import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, RefreshCw, ChevronDown, Copy, Check } from 'lucide-react';

interface Model {
  id: string;
  owned_by: string;
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
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('/v1/models')
      .then(res => res.json())
      .then(data => {
        const allFetched = data.data;
        if (allFetched.length > 0) {
          const uniqueServices = Array.from(new Set(allFetched.map((m: Model) => m.owned_by))) as string[];
          setServices(uniqueServices);

          let currentService = localStorage.getItem('qar_selected_service');
          if (!currentService || !uniqueServices.includes(currentService)) {
             currentService = uniqueServices[0];
             setSelectedService(currentService);
             localStorage.setItem('qar_selected_service', currentService);
          } else {
             setSelectedService(currentService);
          }

          fetch(`/v1/models/${currentService}`)
            .then(r => r.json())
            .then(serviceData => {
               const serviceModels = serviceData.data;
               setModels(serviceModels);
               
               const currentModel = localStorage.getItem('qar_selected_model');
               const fullId = `${currentService}/${currentModel}`;
               const isValidModel = serviceModels.some((m: Model) => m.id === fullId);
               
               if (!isValidModel || !currentModel) {
                  const firstModelId = serviceModels[0]?.id;
                  const firstModelName = firstModelId ? firstModelId.split('/')[1] : '';
                  setSelectedModel(firstModelName);
                  localStorage.setItem('qar_selected_model', firstModelName);
               } else {
                  setSelectedModel(currentModel);
               }
            });
        }
      });
  }, []);

  const handleServiceChange = (service: string) => {
    setSelectedService(service);
    localStorage.setItem('qar_selected_service', service);
    
    fetch(`/v1/models/${service}`)
      .then(res => res.json())
      .then(data => {
        const serviceModels = data.data;
        setModels(serviceModels);
        if (serviceModels.length > 0) {
          const firstModelName = serviceModels[0].id.split('/')[1];
          setSelectedModel(firstModelName);
          localStorage.setItem('qar_selected_model', firstModelName);
        } else {
          setSelectedModel('');
          localStorage.setItem('qar_selected_model', '');
        }
      });
  };

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    localStorage.setItem('qar_selected_model', model);
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
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-brand-400" />
          <span className="font-medium">Chat Tester</span>
        </div>
        <div className="flex items-center gap-2">
          <select 
            className="input-field py-1 text-sm bg-slate-800"
            value={selectedService}
            onChange={(e) => handleServiceChange(e.target.value)}
          >
            {services.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input
            list="model-options"
            className="input-field py-1 text-sm bg-slate-800 w-64 pr-8"
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            placeholder="Type to search model..."
          />
          <datalist id="model-options">
            {models.map(m => {
              const modelName = m.id.split('/')[1];
              return <option key={modelName} value={modelName} />;
            })}
          </datalist>
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
