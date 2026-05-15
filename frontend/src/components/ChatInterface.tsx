import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, RefreshCw, Copy, Check } from 'lucide-react';

interface Model {
  id: string;
  owned_by: string;
  name?: string;
  description?: string;
  is_free?: boolean;
  pricing?: Record<string, string> | null;
  output_modalities?: string[];
  capabilities?: {
    supports_audio_output?: boolean;
  } | null;
}

interface ProviderInfo {
  id?: string;
  type?: string;
  actual_model?: string;
}

interface AudioPayload {
  data?: string;
  format?: string;
  transcript?: string;
  sample_rate_hz?: number;
}

interface ChatMessage {
  role: string;
  content?: string | null;
  audio?: AudioPayload | null;
  reasoning?: string | null;
  refusal?: string | null;
}

interface GeminiSpeakerConfig {
  speaker: string;
  voice: string;
}

type GeminiSafetySettings = Record<string, string>;

const GEMINI_VOICES = [
  'Kore',
  'Puck',
  'Zephyr',
  'Charon',
  'Aoede',
  'Autonoe',
  'Enceladus',
  'Iapetus',
  'Schedar',
  'Sulafat',
];

const GEMINI_SAFETY_CATEGORIES = [
  { key: 'HARM_CATEGORY_HARASSMENT', label: 'Harassment' },
  { key: 'HARM_CATEGORY_HATE_SPEECH', label: 'Hate speech' },
  { key: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', label: 'Sexually explicit' },
  { key: 'HARM_CATEGORY_DANGEROUS_CONTENT', label: 'Dangerous content' },
];

const GEMINI_SAFETY_THRESHOLDS = [
  { value: '', label: 'Default safety' },
  { value: 'BLOCK_LOW_AND_ABOVE', label: 'Block low and above' },
  { value: 'BLOCK_MEDIUM_AND_ABOVE', label: 'Block medium and above' },
  { value: 'BLOCK_ONLY_HIGH', label: 'Block only high' },
  { value: 'BLOCK_NONE', label: 'Block none' },
  { value: 'OFF', label: 'Off' },
];

const createInitialGeminiSafetySettings = (): GeminiSafetySettings => (
  GEMINI_SAFETY_CATEGORIES.reduce<GeminiSafetySettings>((settings, category) => {
    settings[category.key] = '';
    return settings;
  }, {})
);

const ChatInterface: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [services, setServices] = useState<string[]>([]);
  const [selectedService, setSelectedService] = useState(localStorage.getItem('qar_selected_service') || '');
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('qar_selected_model') || '');
  const [copied, setCopied] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastProvider, setLastProvider] = useState<ProviderInfo | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const modelPickerRef = useRef<HTMLDivElement>(null);
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);
  const [geminiCachedContent, setGeminiCachedContent] = useState('');
  const [geminiServiceTier, setGeminiServiceTier] = useState('standard');
  const [geminiThinkingLevel, setGeminiThinkingLevel] = useState('');
  const [geminiIncludeThoughts, setGeminiIncludeThoughts] = useState(false);
  const [geminiAudioEnabled, setGeminiAudioEnabled] = useState(false);
  const [geminiVoice, setGeminiVoice] = useState('Kore');
  const [geminiSafetySettings, setGeminiSafetySettings] = useState<GeminiSafetySettings>(() => createInitialGeminiSafetySettings());
  const [geminiMultiSpeakerEnabled, setGeminiMultiSpeakerEnabled] = useState(false);
  const [geminiSpeakers, setGeminiSpeakers] = useState<GeminiSpeakerConfig[]>([
    { speaker: 'Speaker1', voice: 'Kore' },
    { speaker: 'Speaker2', voice: 'Puck' },
  ]);

  const trimServicePrefix = (service: string, modelId: string) => {
    const prefix = `${service}/`;
    return modelId.startsWith(prefix) ? modelId.slice(prefix.length) : modelId;
  };

  const formatServiceLabel = (service: string) => service.replace(/_/g, ' ');

  const selectedModelInfo = models.find((model) => trimServicePrefix(selectedService, model.id) === selectedModel);
  const geminiSupportsAudioOutput = selectedService === 'gemini' && Boolean(
    selectedModelInfo?.capabilities?.supports_audio_output || selectedModelInfo?.output_modalities?.includes('audio')
  );
  const freeModelCount = models.filter((model) => model.is_free).length;
  const filteredModels = models
    .filter((model) => {
      const modelName = trimServicePrefix(selectedService, model.id).toLowerCase();
      const displayName = (model.name || '').toLowerCase();
      const query = selectedModel.toLowerCase();
      return !query || modelName.includes(query) || displayName.includes(query);
    })
    .slice(0, 40);

  const loadModelsForService = async (service: string, preferredModel?: string) => {
    const response = await fetch(`/v1/models/${service}`);
    const data = await response.json();
    const serviceModels = data.data as Model[];
    setModels(serviceModels);

    const preferred = preferredModel || localStorage.getItem('qar_selected_model') || '';
    const preferredFullId = `${service}/${preferred}`;
    const hasPreferred = preferred && serviceModels.some((model) => model.id === preferredFullId);

    if (hasPreferred) {
      setSelectedModel(preferred);
      localStorage.setItem('qar_selected_model', preferred);
      return;
    }

    const firstModelId = serviceModels[0]?.id;
    const firstModelName = firstModelId ? trimServicePrefix(service, firstModelId) : '';
    setSelectedModel(firstModelName);
    localStorage.setItem('qar_selected_model', firstModelName);
  };

  const handleModelPick = (modelId: string) => {
    const modelName = trimServicePrefix(selectedService, modelId);
    setSelectedModel(modelName);
    localStorage.setItem('qar_selected_model', modelName);
    setIsModelPickerOpen(false);
  };

  useEffect(() => {
    const loadInitialModels = async () => {
      const response = await fetch('/v1/models');
      const data = await response.json();
      const allFetched = data.data as Model[];
      if (!allFetched.length) {
        return;
      }

      const uniqueServices = Array.from(new Set(allFetched.map((model) => model.owned_by))) as string[];
      setServices(uniqueServices);

      let currentService = localStorage.getItem('qar_selected_service') || '';
      if (!currentService || !uniqueServices.includes(currentService)) {
        currentService = uniqueServices[0];
        localStorage.setItem('qar_selected_service', currentService);
      }

      setSelectedService(currentService);
      await loadModelsForService(currentService);
    };

    void loadInitialModels();
  }, []);

  useEffect(() => {
    const handleOutsideClick = (event: MouseEvent) => {
      if (modelPickerRef.current && !modelPickerRef.current.contains(event.target as Node)) {
        setIsModelPickerOpen(false);
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const handleServiceChange = (service: string) => {
    setSelectedService(service);
    localStorage.setItem('qar_selected_service', service);

    void loadModelsForService(service, '');
  };

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    localStorage.setItem('qar_selected_model', model);
    setIsModelPickerOpen(true);
  };

  const handleCopyModel = () => {
    const combinedModel = `${selectedService}/${selectedModel}`;
    if (!combinedModel) return;
    navigator.clipboard.writeText(combinedModel);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!geminiSupportsAudioOutput && geminiAudioEnabled) {
      setGeminiAudioEnabled(false);
    }
  }, [geminiSupportsAudioOutput, geminiAudioEnabled]);

  useEffect(() => {
    if (!geminiAudioEnabled && geminiMultiSpeakerEnabled) {
      setGeminiMultiSpeakerEnabled(false);
    }
  }, [geminiAudioEnabled, geminiMultiSpeakerEnabled]);

  const updateGeminiSpeaker = (index: number, field: keyof GeminiSpeakerConfig, value: string) => {
    setGeminiSpeakers((current) => current.map((speaker, speakerIndex) => (
      speakerIndex === index ? { ...speaker, [field]: value } : speaker
    )));
  };

  const decodeBase64 = (value: string) => {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  };

  const encodeBase64 = (bytes: Uint8Array) => {
    let binary = '';
    const chunkSize = 0x8000;
    for (let index = 0; index < bytes.length; index += chunkSize) {
      const chunk = bytes.subarray(index, index + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  };

  const wrapPcm16AsWav = (base64Data: string, sampleRate: number) => {
    const pcmBytes = decodeBase64(base64Data);
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const writeAscii = (offset: number, text: string) => {
      for (let index = 0; index < text.length; index += 1) {
        view.setUint8(offset + index, text.charCodeAt(index));
      }
    };

    const channelCount = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * channelCount * (bitsPerSample / 8);
    const blockAlign = channelCount * (bitsPerSample / 8);

    writeAscii(0, 'RIFF');
    view.setUint32(4, 36 + pcmBytes.length, true);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channelCount, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeAscii(36, 'data');
    view.setUint32(40, pcmBytes.length, true);

    const wavBytes = new Uint8Array(44 + pcmBytes.length);
    wavBytes.set(new Uint8Array(header), 0);
    wavBytes.set(pcmBytes, 44);

    return encodeBase64(wavBytes);
  };

  const buildAudioPreviewSrc = (audio?: AudioPayload | null) => {
    if (!audio?.data) return null;
    const format = (audio.format || '').toLowerCase();
    const mimeTypeByFormat: Record<string, string> = {
      wav: 'audio/wav',
      wave: 'audio/wav',
      mp3: 'audio/mpeg',
      mpeg: 'audio/mpeg',
      ogg: 'audio/ogg',
      webm: 'audio/webm',
    };

    if (format === 'pcm16' || format === 'pcm') {
      const wavBase64 = wrapPcm16AsWav(audio.data, audio.sample_rate_hz || 24000);
      return `data:audio/wav;base64,${wavBase64}`;
    }

    const mimeType = mimeTypeByFormat[format];
    if (!mimeType) return null;
    return `data:${mimeType};base64,${audio.data}`;
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input || !selectedModel) return;

    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const requestBody: Record<string, unknown> = {
        model: `${selectedService}/${selectedModel}`,
        messages: newMessages.map((message) => ({ role: message.role, content: message.content })),
        stream: false,
      };

      if (selectedService === 'gemini') {
        const googleOptions: Record<string, unknown> = {};

        if (geminiCachedContent.trim()) {
          googleOptions.cached_content = geminiCachedContent.trim();
        }

        if (geminiServiceTier !== 'standard') {
          googleOptions.service_tier = geminiServiceTier;
        }

        if (geminiIncludeThoughts || geminiThinkingLevel) {
          const thinkingConfig: Record<string, unknown> = {};
          if (geminiIncludeThoughts) {
            thinkingConfig.includeThoughts = true;
          }
          if (geminiThinkingLevel) {
            thinkingConfig.thinkingLevel = geminiThinkingLevel.toUpperCase();
          }
          googleOptions.thinking_config = thinkingConfig;
        }

        const safetySettings = GEMINI_SAFETY_CATEGORIES
          .map((category) => ({
            category: category.key,
            threshold: geminiSafetySettings[category.key],
          }))
          .filter((setting) => setting.threshold);

        if (safetySettings.length > 0) {
          googleOptions.safety_settings = safetySettings;
        }

        if (geminiAudioEnabled && geminiSupportsAudioOutput) {
          requestBody.modalities = ['audio'];
          const audioPayload: Record<string, unknown> = {
            format: 'wav',
            language: 'en-US',
          };

          if (geminiMultiSpeakerEnabled) {
            audioPayload.speakers = geminiSpeakers.filter((speaker) => speaker.speaker.trim() && speaker.voice.trim());
          } else {
            audioPayload.voice = geminiVoice;
          }

          requestBody.audio = audioPayload;
        }

        if (Object.keys(googleOptions).length > 0) {
          requestBody.extra_body = { google: googleOptions };
        }
      }

      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      const data = await response.json();
      setLastProvider(data.provider ?? null);
      if (data.choices && data.choices.length > 0) {
        setMessages([...newMessages, data.choices[0].message]);
      } else if (data.error) {
        setMessages([...newMessages, { role: 'assistant', content: `Error: ${data.error.message || JSON.stringify(data.error)}` }]);
      } else {
        setMessages([...newMessages, { role: 'assistant', content: `Unexpected response: ${JSON.stringify(data)}` }]);
      }
    } catch (error) {
      setMessages([...newMessages, { role: 'assistant', content: 'Connection failed.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] glass-card overflow-hidden">
      <div className="p-4 border-b border-white/10 flex justify-between items-center bg-white/5">
        <div>
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-brand-400" />
            <span className="font-medium">Chat Tester</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            {lastProvider?.id ? `Last routed via ${lastProvider.id}${lastProvider.actual_model ? ` · ${lastProvider.actual_model}` : ''}` : 'No routed provider yet'}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {models.length ? `${models.length} models loaded${freeModelCount ? ` · ${freeModelCount} free` : ''}` : 'No models loaded'}
            {selectedModelInfo?.is_free ? ' · selected model is free' : ''}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <select 
            className="input-field py-1 text-sm bg-slate-800"
            value={selectedService}
            onChange={(e) => handleServiceChange(e.target.value)}
          >
            {services.map(s => (
              <option key={s} value={s}>{formatServiceLabel(s)}</option>
            ))}
          </select>
          <div ref={modelPickerRef} className="relative w-80">
            <input
              className="input-field py-1 text-sm bg-slate-800 w-full"
              value={selectedModel}
              onFocus={() => setIsModelPickerOpen(true)}
              onChange={(e) => handleModelChange(e.target.value)}
              placeholder="Type to search model..."
            />
            {isModelPickerOpen && (
              <div className="absolute top-[calc(100%+0.5rem)] z-20 w-full overflow-hidden rounded-xl border border-white/10 bg-slate-950/95 shadow-2xl backdrop-blur-md">
                <div className="max-h-80 overflow-y-auto">
                  {filteredModels.length ? filteredModels.map((model) => {
                    const modelName = trimServicePrefix(selectedService, model.id);
                    const isSelected = modelName === selectedModel;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        className={`w-full border-b border-white/5 px-3 py-2 text-left transition-all last:border-b-0 ${isSelected ? 'bg-brand-500/15' : 'hover:bg-white/5'}`}
                        onMouseDown={(event) => {
                          event.preventDefault();
                          handleModelPick(model.id);
                        }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="truncate text-sm text-slate-100">{modelName}</span>
                          {model.is_free ? (
                            <span className="rounded-full border border-emerald-400/30 bg-emerald-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-300">
                              Free
                            </span>
                          ) : null}
                        </div>
                        {model.description ? (
                          <p className="mt-1 line-clamp-2 text-xs text-slate-500">{model.description}</p>
                        ) : null}
                      </button>
                    );
                  }) : (
                    <div className="px-3 py-4 text-sm text-slate-500">No models match this search.</div>
                  )}
                </div>
              </div>
            )}
          </div>
          <button 
            onClick={handleCopyModel}
            className="p-2 hover:bg-white/10 rounded-lg transition-all text-slate-400 hover:text-white border border-white/10"
            title="Copy Model ID"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {selectedService === 'gemini' && (
        <div className="border-b border-white/10 bg-white/[0.03] px-4 py-3">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
            <input
              className="input-field py-2 text-sm bg-slate-800"
              value={geminiCachedContent}
              onChange={(e) => setGeminiCachedContent(e.target.value)}
              placeholder="Gemini cachedContent (optional)"
            />
            <select
              className="input-field py-2 text-sm bg-slate-800"
              value={geminiServiceTier}
              onChange={(e) => setGeminiServiceTier(e.target.value)}
            >
              <option value="standard">Gemini tier: standard</option>
              <option value="flex">Gemini tier: flex</option>
              <option value="priority">Gemini tier: priority</option>
            </select>
            <select
              className="input-field py-2 text-sm bg-slate-800"
              value={geminiThinkingLevel}
              onChange={(e) => setGeminiThinkingLevel(e.target.value)}
            >
              <option value="">Thinking level: default</option>
              <option value="minimal">Thinking: minimal</option>
              <option value="low">Thinking: low</option>
              <option value="medium">Thinking: medium</option>
              <option value="high">Thinking: high</option>
            </select>
            <label className="flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900/70 px-3 py-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={geminiIncludeThoughts}
                onChange={(e) => setGeminiIncludeThoughts(e.target.checked)}
              />
              Include Gemini thoughts
            </label>
            {geminiSupportsAudioOutput ? (
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900/70 px-3 py-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={geminiAudioEnabled}
                    onChange={(e) => setGeminiAudioEnabled(e.target.checked)}
                  />
                  Audio output
                </label>
                <select
                  className="input-field py-2 text-sm bg-slate-800"
                  value={geminiVoice}
                  onChange={(e) => setGeminiVoice(e.target.value)}
                  disabled={!geminiAudioEnabled}
                >
                  {GEMINI_VOICES.map((voice) => (
                    <option key={voice} value={voice}>{voice}</option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="rounded-xl border border-white/10 bg-slate-900/70 px-3 py-2 text-xs text-slate-500">
                Selected Gemini model does not advertise audio output.
              </div>
            )}
          </div>
          <div className="mt-3 rounded-2xl border border-white/10 bg-slate-950/60 p-3">
            <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm font-medium text-slate-100">Advanced Gemini</p>
                <p className="text-xs text-slate-500">Safety thresholds and multi-speaker TTS stay scoped to Gemini requests only.</p>
              </div>
              {geminiSupportsAudioOutput ? (
                <label className="flex items-center gap-2 rounded-xl border border-white/10 bg-slate-900/70 px-3 py-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={geminiMultiSpeakerEnabled}
                    onChange={(e) => setGeminiMultiSpeakerEnabled(e.target.checked)}
                    disabled={!geminiAudioEnabled}
                  />
                  Multi-speaker TTS
                </label>
              ) : null}
            </div>

            <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
              <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Safety Settings</p>
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                  {GEMINI_SAFETY_CATEGORIES.map((category) => (
                    <label key={category.key} className="space-y-1 text-xs text-slate-400">
                      <span>{category.label}</span>
                      <select
                        className="input-field py-2 text-sm bg-slate-800"
                        value={geminiSafetySettings[category.key]}
                        onChange={(e) => setGeminiSafetySettings((current) => ({
                          ...current,
                          [category.key]: e.target.value,
                        }))}
                      >
                        {GEMINI_SAFETY_THRESHOLDS.map((threshold) => (
                          <option key={threshold.value || 'default'} value={threshold.value}>{threshold.label}</option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">TTS Speakers</p>
                {geminiSupportsAudioOutput ? (
                  geminiMultiSpeakerEnabled ? (
                    <div className="space-y-3">
                      {geminiSpeakers.map((speaker, index) => (
                        <div key={`${speaker.voice}-${index}`} className="grid grid-cols-1 gap-2 md:grid-cols-2">
                          <input
                            className="input-field py-2 text-sm bg-slate-800"
                            value={speaker.speaker}
                            onChange={(e) => updateGeminiSpeaker(index, 'speaker', e.target.value)}
                            placeholder={`Speaker ${index + 1} name`}
                          />
                          <select
                            className="input-field py-2 text-sm bg-slate-800"
                            value={speaker.voice}
                            onChange={(e) => updateGeminiSpeaker(index, 'voice', e.target.value)}
                          >
                            {GEMINI_VOICES.map((voice) => (
                              <option key={voice} value={voice}>{voice}</option>
                            ))}
                          </select>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">Single-speaker mode is active. Enable multi-speaker TTS to send named speakers and voices.</p>
                  )
                ) : (
                  <p className="text-sm text-slate-500">This model is not advertising Gemini audio output, so TTS speaker controls stay disabled.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((m, i) => {
          const audioPreviewSrc = buildAudioPreviewSrc(m.audio);

          return (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] p-3 rounded-2xl flex gap-3 ${
                m.role === 'user' ? 'bg-brand-600' : 'bg-white/10 border border-white/5'
              }`}>
                <div className="shrink-0">
                  {m.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5 text-brand-400" />}
                </div>
                <div className="space-y-2">
                  <p className="text-sm whitespace-pre-wrap">{m.content || m.refusal || (m.audio ? 'Audio response returned.' : '')}</p>
                  {m.reasoning ? (
                    <p className="text-xs whitespace-pre-wrap rounded-lg bg-black/20 px-3 py-2 text-slate-400">
                      Reasoning: {m.reasoning}
                    </p>
                  ) : null}
                  {m.audio?.transcript ? (
                    <p className="text-xs whitespace-pre-wrap rounded-lg bg-black/20 px-3 py-2 text-slate-300">
                      Transcript: {m.audio.transcript}
                    </p>
                  ) : null}
                  {m.audio ? (
                    <div className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs text-slate-400">
                      <p>Audio format: {m.audio.format || 'unknown'}</p>
                      {m.audio.sample_rate_hz ? <p className="mt-1">Sample rate: {m.audio.sample_rate_hz} Hz</p> : null}
                      {audioPreviewSrc ? (
                        <audio className="mt-2 w-full" controls src={audioPreviewSrc} />
                      ) : (
                        <p className="mt-1">Browser preview unavailable for this audio payload.</p>
                      )}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white/10 p-3 rounded-2xl animate-pulse flex gap-2">
              <Bot className="w-5 h-5 text-brand-400" />
              <div className="flex gap-1 items-center">
                <div className="w-1 h-1 bg-brand-400 rounded-full animate-bounce" />
                <div className="w-1 h-1 bg-brand-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                <div className="w-1 h-1 bg-brand-400 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          </div>
        )}
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-50">
            <Bot className="w-12 h-12 mb-4" />
            <p>Select a model and start a conversation</p>
          </div>
        )}
      </div>

      <form onSubmit={sendMessage} className="p-4 border-t border-white/10 bg-white/5 flex gap-2">
        <input 
          className="flex-1 input-field"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button type="submit" className="btn-primary p-2 flex items-center justify-center aspect-square" disabled={loading}>
          {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
