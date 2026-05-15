import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, RefreshCw, Copy, Check } from 'lucide-react';

interface Model {
  id: string;
  owned_by: string;
  name?: string;
  description?: string;
  is_free?: boolean;
  pricing?: Record<string, string> | null;
}

interface ProviderInfo {
  id?: string;
  type?: string;
  actual_model?: string;
}

const ChatInterface: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [services, setServices] = useState<string[]>([]);
  const [selectedService, setSelectedService] = useState(localStorage.getItem('qar_selected_service') || '');
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('qar_selected_model') || '');
  const [copied, setCopied] = useState(false);
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastProvider, setLastProvider] = useState<ProviderInfo | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const modelPickerRef = useRef<HTMLDivElement>(null);
  const [isModelPickerOpen, setIsModelPickerOpen] = useState(false);

  const trimServicePrefix = (service: string, modelId: string) => {
    const prefix = `${service}/`;
    return modelId.startsWith(prefix) ? modelId.slice(prefix.length) : modelId;
  };

  const formatServiceLabel = (service: string) => service.replace(/_/g, ' ');

  const selectedModelInfo = models.find((model) => trimServicePrefix(selectedService, model.id) === selectedModel);
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

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input || !selectedModel) return;

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: `${selectedService}/${selectedModel}`,
          messages: newMessages,
          stream: false
        })
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

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-2xl flex gap-3 ${
              m.role === 'user' ? 'bg-brand-600' : 'bg-white/10 border border-white/5'
            }`}>
              <div className="shrink-0">
                {m.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5 text-brand-400" />}
              </div>
              <p className="text-sm whitespace-pre-wrap">{m.content}</p>
            </div>
          </div>
        ))}
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
