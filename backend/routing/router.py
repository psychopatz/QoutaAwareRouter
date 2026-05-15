from .model_aliases import ModelAliasManager, Candidate
from .traffic import traffic_manager, TrafficLog
from ..schemas import ChatCompletionRequest, ChatCompletionResponse
from ..errors import GatewayError, ErrorType
from ..logging_config import logger
import time
import uuid

class Router:
    def __init__(self, alias_manager: ModelAliasManager, routing_config: dict):
        self.alias_manager = alias_manager
        self.routing_config = routing_config
        self.max_retries = routing_config.get("max_retries_total", 2)

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

    async def _route_internal(self, request: ChatCompletionRequest, log_id: str, start_time: float) -> ChatCompletionResponse:
        # Handle prefixed model names (e.g., 'ollama/llama3')
        requested_model = request.model
        required_service = None
        if "/" in requested_model:
            required_service, requested_model = requested_model.split("/", 1)
            logger.info(f"Prefix detected: service={required_service}, model={requested_model}")

        candidates = self.alias_manager.resolve(requested_model)
        
        if not candidates:
            # Fallback: check if any provider directly supports this model
            from ..providers.registry import registry
            for p_id, provider in registry.get_all_instances().items():
                if provider.enabled:
                    # If service prefix is present, only consider matching providers
                    if required_service and provider.type != required_service:
                        continue
                    candidates.append(Candidate(provider=p_id, model=requested_model, priority=1))
            
            if not candidates:
                msg = f"Model '{requested_model}' not found"
                if required_service: msg += f" for service '{required_service}'"
                raise GatewayError(msg, type=ErrorType.INVALID_MODEL, status_code=400)

        # Filter and sort candidates based on strategy
        # Phase 2: priority_failover (implied by candidate order)
        
        errors = []
        for candidate in candidates:
            # Ensure the service restriction is checked again internally just in case candidates came from ALIAS
            provider = registry.get_instance(candidate.provider)
            if not provider or not provider.enabled:
                logger.debug(f"Provider {candidate.provider} is disabled or not found, skipping")
                continue
                
            if required_service and provider.type != required_service:
                logger.debug(f"Provider {candidate.provider} does not match required service {required_service}, skipping")
                continue

            # TODO: Check health, cooldowns, concurrency limits (Phase 2/6)
            
            # Prepare request with actual provider model
            # We use candidate.model instead of requested_model just in case alias specifies a different name
            provider_request = request.model_copy(update={"model": candidate.model})
            
            # Dynamic Key Logic (Phase 4 integration)
            from ..storage.sqlite import storage
            dynamic_keys = storage.get_keys_by_service(provider.type)
            active_keys = [k for k in dynamic_keys if k.status == "active"]
            
            # If we have dynamic keys, use them. Otherwise use provider's static key.
            keys_to_try = active_keys if active_keys else [None]
            
            for d_key in keys_to_try:
                if d_key:
                    # Pass the key to the provider temporarily
                    provider.api_key = d_key.key
                
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
                        status_code=e.status_code,
                        latency_ms=latency,
                        error=e.message
                    ))
                    
                    if e.type == ErrorType.RATE_LIMITED and d_key:
                        logger.warning(f"Key {d_key.id} for {provider.type} is rate limited, marking in DB")
                        storage.update_key_status(d_key.id, "rate_limited")
                        continue # Try next key
                    
                    logger.warning(f"Provider {provider.id} failed: {e.message}")
                    errors.append(e)
                    break # Try next provider candidate
                except Exception as e:
                    logger.error(f"Unexpected error from provider {provider.id}: {e}")
                    errors.append(GatewayError(str(e), provider_id=provider.id))
                    break # Try next provider candidate

        # If we reached here, all candidates failed
        if errors:
            # Return the last relevant error or a generic one
            last_error = errors[-1]
            raise GatewayError(
                f"All candidates for model '{request.model}' failed. Last error: {last_error.message}",
                type=ErrorType.ALL_PROVIDERS_UNAVAILABLE,
                status_code=503
            )
        
        raise GatewayError(f"No healthy providers found for model '{request.model}'", type=ErrorType.ALL_PROVIDERS_UNAVAILABLE, status_code=503)

    async def route_stream(self, request: ChatCompletionRequest):
        # Placeholder for streaming (Phase 3)
        pass
