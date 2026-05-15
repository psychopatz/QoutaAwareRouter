import React, { useState, useEffect } from 'react';
import { Plus, Trash2, RefreshCw, Key, ShieldCheck, AlertCircle, Copy, Check, MessageSquare, List, Activity, Settings } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import TrafficLogs from './components/TrafficLogs';

interface ApiKey {
  id: number;
  service: string;
  key: string;
  status: string;
}

const App: React.FC = () => {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKey, setNewKey] = useState({ service: 'ollama', key: '' });
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'keys' | 'chat' | 'logs'>('keys');
  const [copiedBase, setCopiedBase] = useState(false);

  const baseUrl = `http://127.0.0.1:7317/v1`;

  const fetchKeys = async () => {
    try {
      const response = await fetch('/admin/keys');
      const data = await response.json();
      setKeys(data);
    } catch (error) {
      console.error('Failed to fetch keys', error);
    }
  };

  const addKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKey.key) return;
    setLoading(true);
    try {
      await fetch('/admin/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newKey),
      });
      setNewKey({ ...newKey, key: '' });
      fetchKeys();
    } catch (error) {
      console.error('Failed to add key', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteKey = async (id: number) => {
    try {
      await fetch(`/admin/keys/${id}`, { method: 'DELETE' });
      fetchKeys();
    } catch (error) {
      console.error('Failed to delete key', error);
    }
  };

  const testKey = async (id: number) => {
    try {
      const response = await fetch(`/admin/keys/${id}/test`, { method: 'POST' });
      const data = await response.json();
      alert(`Test Result:\n${data.message}`);
      fetchKeys();
    } catch (error) {
      console.error('Failed to test key', error);
      alert('Error: Failed to connect to server');
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(baseUrl);
    setCopiedBase(true);
    setTimeout(() => setCopiedBase(false), 2000);
  };

  useEffect(() => {
    fetchKeys();
    const interval = setInterval(fetchKeys, 5000);
    return () => clearInterval(interval);
  }, []);

  const groupedKeys = keys.reduce((acc, key) => {
    if (!acc[key.service]) acc[key.service] = [];
    acc[key.service].push(key);
    return acc;
  }, {} as Record<string, ApiKey[]>);

  return (
    <div className="max-w-6xl mx-auto p-8">
      <header className="mb-12">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6 mb-8">
          <div className="text-center md:text-left">
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-brand-400 to-brand-600 bg-clip-text text-transparent">
              Quota Aware LLM Router
            </h1>
            <p className="text-slate-400">Manage and cycle your LLM provider API keys seamlessly.</p>
          </div>
          <div className="glass-card px-4 py-2 flex items-center gap-4 bg-white/5 border-emerald-500/20">
            <div>
              <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-0.5">LLM Server Base URL</p>
              <code className="text-brand-400 text-sm font-mono">{baseUrl}</code>
            </div>
            <button 
              onClick={copyToClipboard}
              className="p-2 hover:bg-white/10 rounded-lg transition-all text-slate-400 hover:text-white"
            >
              {copiedBase ? <Check className="w-5 h-5 text-emerald-400" /> : <Copy className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <nav className="flex gap-2 p-1.5 glass-card bg-slate-900/50 w-fit mx-auto md:mx-0">
          <button 
            onClick={() => setActiveTab('keys')}
            className={`flex items-center gap-2 px-6 py-2 rounded-xl transition-all ${activeTab === 'keys' ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/20' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
          >
            <Settings className="w-4 h-4" />
            Keys
          </button>
          <button 
            onClick={() => setActiveTab('chat')}
            className={`flex items-center gap-2 px-6 py-2 rounded-xl transition-all ${activeTab === 'chat' ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/20' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
          >
            <MessageSquare className="w-4 h-4" />
            Chat Test
          </button>
          <button 
            onClick={() => setActiveTab('logs')}
            className={`flex items-center gap-2 px-6 py-2 rounded-xl transition-all ${activeTab === 'logs' ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/20' : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'}`}
          >
            <Activity className="w-4 h-4" />
            Traffic
          </button>
        </nav>
      </header>

      <main className="space-y-12">
        {activeTab === 'keys' && (
          <>
            <section className="glass-card p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <Plus className="w-5 h-5 text-brand-400" />
                Add New API Key
              </h2>
              <form onSubmit={addKey} className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <select 
                  className="input-field"
                  value={newKey.service}
                  onChange={(e) => setNewKey({ ...newKey, service: e.target.value })}
                >
                  <option value="ollama">Ollama Cloud</option>
                  <option value="openrouter">OpenRouter</option>
                  <option value="openai">OpenAI Compatible</option>
                </select>
                <input 
                  type="password"
                  placeholder="Enter API Key"
                  className="input-field"
                  value={newKey.key}
                  onChange={(e) => setNewKey({ ...newKey, key: e.target.value })}
                />
                <button type="submit" className="btn-primary flex items-center justify-center gap-2" disabled={loading}>
                  {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Register Key'}
                </button>
              </form>
            </section>

            <div className="space-y-12">
              {Object.entries(groupedKeys).map(([service, serviceKeys]) => (
                <section key={service}>
                  <h3 className="text-lg font-medium mb-6 flex items-center gap-2 capitalize">
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                    {service} Keys
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {serviceKeys.map((key) => (
                      <div key={key.id} className="glass-card p-5 relative group overflow-hidden">
                        <div className="flex items-start justify-between mb-4">
                          <div className="bg-white/5 p-2 rounded-lg">
                            <Key className="w-5 h-5 text-brand-300" />
                          </div>
                          <div className={`text-xs px-2 py-1 rounded-full flex items-center gap-1 ${
                            key.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                            key.status === 'rate_limited' ? 'bg-amber-500/20 text-amber-400' :
                            'bg-rose-500/20 text-rose-400'
                          }`}>
                            {key.status === 'active' ? <ShieldCheck className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                            {key.status.replace('_', ' ')}
                          </div>
                        </div>
                        <div className="mb-6">
                          <p className="text-[10px] text-slate-500 mb-1 font-bold uppercase tracking-widest">Key Suffix</p>
                          <code className="text-sm font-mono text-slate-300">
                            ••••••••{key.key.slice(-4)}
                          </code>
                        </div>
                        <button 
                          onClick={() => deleteKey(key.id!)}
                          className="absolute bottom-4 right-4 text-slate-500 hover:text-rose-400 p-2 rounded-lg hover:bg-rose-400/10 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => testKey(key.id!)}
                          className="absolute bottom-4 right-14 text-slate-500 hover:text-emerald-400 p-2 rounded-lg hover:bg-emerald-400/10 transition-all opacity-0 group-hover:opacity-100"
                          title="Test Key"
                        >
                          <Activity className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                </section>
              ))}
              {keys.length === 0 && (
                <div className="text-center py-20 bg-white/5 rounded-3xl border border-dashed border-white/10">
                  <p className="text-slate-500">No API keys registered yet.</p>
                </div>
              )}
            </div>
          </>
        )}

        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'logs' && <TrafficLogs />}
      </main>
    </div>
  );
};

export default App;
