from .model_aliases import ModelAliasManager, Candidate
from .traffic import traffic_manager, TrafficLog
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError, ErrorType
from ..logging_config import logger
from ..streaming.control import ProviderStreamControl
import time
import uuid

class Router:
    def __init__(self, alias_manager: ModelAliasManager, routing_config: dict):
        self.alias_manager = alias_manager
        self.routing_config = routing_config
        self.max_retries = routing_config.get("max_retries_total", 2)

    def _is_quota_exhausted_error(self, error: GatewayError) -> bool:
        if error.type == ErrorType.QUOTA_LIMITED:
            return True

        message = (error.message or "").lower()
        quota_markers = (
            "quota",
            "usage limit",
            "weekly usage limit",
            "monthly usage limit",
            "credit balance",
            "insufficient credits",
            "out of credits",
            "upgrade for higher limits",
            "exhausted",
        )
        return any(marker in message for marker in quota_markers)

    def _record_key_attempt(self, storage, dynamic_key, provider, model: str):
        if dynamic_key is None:
            return
        storage.record_key_usage(dynamic_key.id, provider.id, model)

    def _key_candidates(self, storage, provider):
        original_api_key = provider.config.get("api_key")
        dynamic_keys = storage.get_keys_by_service(provider.type)
        active_keys = [key for key in dynamic_keys if key.status == "active"]

        candidates = list(active_keys)
        has_static_fallback = bool(original_api_key) and all(key.key != original_api_key for key in active_keys)
        if has_static_fallback or not candidates:
            candidates.append(None)

        return original_api_key, candidates

    def _apply_key_candidate(self, provider, dynamic_key, original_api_key):
        if dynamic_key is not None:
            provider.config["api_key"] = dynamic_key.key
        elif original_api_key:
            provider.config["api_key"] = original_api_key
        else:
            provider.config.pop("api_key", None)

    def _handle_dynamic_key_error(self, storage, dynamic_key, error: GatewayError) -> bool:
        if dynamic_key is None:
            return False

        if self._is_quota_exhausted_error(error):
            logger.warning(f"Key {dynamic_key.id} is quota exhausted, marking inactive in DB")
            storage.record_key_exhausted(dynamic_key.id, error.message)
            return True

        if error.type == ErrorType.RATE_LIMITED:
            logger.warning(f"Key {dynamic_key.id} is rate limited, marking in DB")
            storage.update_key_status(dynamic_key.id, "rate_limited", last_status_message=error.message)
            return True

        if error.type == ErrorType.AUTH_FAILED:
            logger.warning(f"Key {dynamic_key.id} failed authentication, marking inactive in DB")
            storage.update_key_status(dynamic_key.id, "auth_failed", last_status_message=error.message)

        return False

    async def route(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        start_time = time.time()
        log_id = str(uuid.uuid4())
        
        try:
            response = await self._route_internal(request, log_id, start_time)
            return response
        except Exception as e:
            # Final fallback log for unhandled errors
            latency = (time.time() - start_time) * 1000
            traffic_manager.add_log(TrafficLog(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                method="POST",
                path="/v1/chat/completions",
                model=request.model,
                provider_id=None,
                status_code=500,
                latency_ms=latency,
                error=str(e)
            ))
            raise

    def _split_requested_model(self, requested_model: str):
        required_service = None
        actual_model = requested_model
        if "/" in requested_model:
            required_service, actual_model = requested_model.split("/", 1)
            logger.info(f"Prefix detected: service={required_service}, model={actual_model}")
        return required_service, actual_model

    async def _resolve_candidates(self, requested_model: str, required_service: str = None):
        from ..providers.registry import registry

        candidates = self.alias_manager.resolve(requested_model)

        if not candidates:
            for p_id, provider in registry.get_all_instances().items():
                if not provider.enabled:
                    continue
                if required_service and provider.type != required_service:
                    continue
                candidates.append(Candidate(provider=p_id, model=requested_model, priority=1))

        return candidates

    async def _filter_candidates_for_request(self, request: ChatCompletionRequest, candidates, required_service: str = None, require_streaming: bool = False):
        from ..providers.registry import registry

        supported_candidates = []
        unsupported_errors = []

        for candidate in candidates:
            provider = registry.get_instance(candidate.provider)
            if not provider or not provider.enabled:
                logger.debug(f"Provider {candidate.provider} is disabled or not found, skipping")
                continue

            if require_streaming and not provider.supports_streaming:
                logger.debug(f"Provider {candidate.provider} does not support streaming, skipping")
                continue

            if required_service and provider.type != required_service:
                logger.debug(f"Provider {candidate.provider} does not match required service {required_service}, skipping")
                continue

            provider_request = request.model_copy(update={"model": candidate.model})

            try:
                await provider.validate_model_request(candidate.model, provider_request)
            except GatewayError as e:
                if e.type == ErrorType.UNSUPPORTED_FEATURE:
                    logger.info(f"Skipping provider={provider.id}, model={candidate.model}: {e.message}")
                    unsupported_errors.append(e)
                    continue
                raise

            supported_candidates.append((candidate, provider, provider_request))

        return supported_candidates, unsupported_errors

    async def _route_internal(self, request: ChatCompletionRequest, log_id: str, start_time: float) -> ChatCompletionResponse:
        requested_model = request.model
        required_service, resolved_model = self._split_requested_model(requested_model)
        candidates = await self._resolve_candidates(resolved_model, required_service)

        if not candidates:
            msg = f"Model '{resolved_model}' not found"
            if required_service:
                msg += f" for service '{required_service}'"
            raise GatewayError(msg, type=ErrorType.INVALID_MODEL, status_code=400)

        routed_candidates, unsupported_errors = await self._filter_candidates_for_request(
            request,
            candidates,
            required_service=required_service,
        )

        errors = []
        for candidate, provider, provider_request in routed_candidates:
            # TODO: Check health, cooldowns, concurrency limits (Phase 2/6)

            # Dynamic Key Logic (Phase 4 integration)
            from ..storage.sqlite import storage
            original_api_key, keys_to_try = self._key_candidates(storage, provider)

            try:
                for d_key in keys_to_try:
                    self._apply_key_candidate(provider, d_key, original_api_key)
                    if d_key:
                        self._record_key_attempt(storage, d_key, provider, candidate.model)

                    try:
                        logger.info(f"Routing request to provider={provider.id}, model={candidate.model} using key_id={d_key.id if d_key else 'default'}")
                        response = await provider.chat_completion(provider_request)
                        latency = (time.time() - start_time) * 1000

                        traffic_manager.add_log(TrafficLog(
                            id=str(uuid.uuid4()),
                            timestamp=time.time(),
                            method="POST",
                            path="/v1/chat/completions",
                            model=request.model,
                            provider_id=provider.id,
                            key_id=d_key.id if d_key else None,
                            key_suffix=d_key.key[-4:] if d_key else None,
                            status_code=200,
                            latency_ms=latency
                        ))

                        response.model = request.model
                        return response
                    except GatewayError as e:
                        latency = (time.time() - start_time) * 1000
                        traffic_manager.add_log(TrafficLog(
                            id=str(uuid.uuid4()),
                            timestamp=time.time(),
                            method="POST",
                            path="/v1/chat/completions",
                            model=request.model,
                            provider_id=provider.id,
                            key_id=d_key.id if d_key else None,
                            key_suffix=d_key.key[-4:] if d_key else None,
                            status_code=e.status_code,
                            latency_ms=latency,
                            error=e.message
                        ))

                        if self._handle_dynamic_key_error(storage, d_key, e):
                            continue

                        logger.warning(f"Provider {provider.id} failed: {e.message}")
                        errors.append(e)
                        break
                    except Exception as e:
                        logger.error(f"Unexpected error from provider {provider.id}: {e}")
                        errors.append(GatewayError(str(e), provider_id=provider.id))
                        break
            finally:
                self._apply_key_candidate(provider, None, original_api_key)

        # If we reached here, all candidates failed
        if errors:
            # Return the last relevant error or a generic one
            last_error = errors[-1]
            if last_error.type == ErrorType.UNSUPPORTED_FEATURE:
                raise last_error
            raise GatewayError(
                f"All candidates for model '{request.model}' failed. Last error: {last_error.message}",
                type=ErrorType.ALL_PROVIDERS_UNAVAILABLE,
                status_code=503
            )

        if unsupported_errors:
            raise unsupported_errors[-1]
        
        raise GatewayError(f"No healthy providers found for model '{request.model}'", type=ErrorType.ALL_PROVIDERS_UNAVAILABLE, status_code=503)

    async def iter_stream(self, request: ChatCompletionRequest, on_provider_selected=None, stream_control: ProviderStreamControl = None):
        import json

        requested_model = request.model
        required_service, resolved_model = self._split_requested_model(requested_model)
        candidates = await self._resolve_candidates(resolved_model, required_service)

        if not candidates:
            msg = f"Model '{resolved_model}' not found"
            if required_service:
                msg += f" for service '{required_service}'"
            raise GatewayError(msg, type=ErrorType.INVALID_MODEL, status_code=400)

        routed_candidates, unsupported_errors = await self._filter_candidates_for_request(
            request,
            candidates,
            required_service=required_service,
            require_streaming=True,
        )

        if not routed_candidates and unsupported_errors:
            raise unsupported_errors[-1]
                
        async def stream_generator():
            nonlocal request
            start_time = time.time()
            errors = []
            
            for candidate, provider, provider_request in routed_candidates:
                
                from ..storage.sqlite import storage
                original_api_key, keys_to_try = self._key_candidates(storage, provider)

                try:
                    for d_key in keys_to_try:
                        self._apply_key_candidate(provider, d_key, original_api_key)
                        if d_key:
                            self._record_key_attempt(storage, d_key, provider, candidate.model)

                        chunks_yielded = False
                        log_id_iter = str(uuid.uuid4())
                        try:
                            logger.info(f"Stream routing to provider={provider.id}, model={candidate.model}")

                            if on_provider_selected is not None:
                                on_provider_selected(
                                    {
                                        "id": provider.id,
                                        "type": provider.type,
                                        "actual_model": candidate.model,
                                    }
                                )

                            async for chunk in provider.stream_chat_completion(provider_request, stream_control=stream_control):
                                chunks_yielded = True
                                if chunk:
                                    yield chunk

                            traffic_manager.add_log(TrafficLog(
                                id=log_id_iter,
                                timestamp=time.time(),
                                method="POST",
                                path="/v1/chat/completions/stream",
                                model=request.model,
                                provider_id=provider.id,
                                key_id=d_key.id if d_key else None,
                                key_suffix=d_key.key[-4:] if d_key else None,
                                status_code=200,
                                latency_ms=(time.time() - start_time) * 1000
                            ))
                            return

                        except Exception as general_e:
                            if stream_control is not None and stream_control.cancelled:
                                return

                            e = general_e if isinstance(general_e, GatewayError) else GatewayError(str(general_e), provider_id=provider.id)

                            if chunks_yielded:
                                logger.error(f"Stream broken mid-way: {e}")
                                err_payload = json.dumps({"error": {"message": f"Stream interrupted: {e.message}"}})
                                yield f"data: {err_payload}\n\n".encode()
                                return

                            traffic_manager.add_log(TrafficLog(
                                id=log_id_iter,
                                timestamp=time.time(),
                                method="POST",
                                path="/v1/chat/completions/stream",
                                model=request.model,
                                provider_id=provider.id,
                                key_id=d_key.id if d_key else None,
                                key_suffix=d_key.key[-4:] if d_key else None,
                                status_code=e.status_code,
                                latency_ms=(time.time() - start_time) * 1000,
                                error=e.message
                            ))

                            if self._handle_dynamic_key_error(storage, d_key, e):
                                continue

                            errors.append(e)
                            break
                finally:
                    self._apply_key_candidate(provider, None, original_api_key)

            last_err = errors[-1].message if errors else "No healthy streaming providers found"
            err_payload = json.dumps({"error": {"message": f"All providers failed. Last error: {last_err}"}})
            yield f"data: {err_payload}\n\n".encode()

        return stream_generator()

    async def route_stream(self, request: ChatCompletionRequest):
        from fastapi.responses import StreamingResponse

        stream = await self.iter_stream(request)
        return StreamingResponse(stream, media_type="text/event-stream")
