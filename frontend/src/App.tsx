import React, { useState, useEffect } from 'react';
import { Plus, Trash2, RefreshCw, Key, ShieldCheck, AlertCircle, Copy, Check, MessageSquare, Activity, Settings } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import Modal from './components/Modal';
import TrafficLogs from './components/TrafficLogs';

interface ApiKey {
  id: number;
  service: string;
  key: string;
  status: string;
  last_used?: string | null;
  cooldown_until?: string | null;
  request_count: number;
  last_used_provider_id?: string | null;
  last_used_model?: string | null;
  last_status_message?: string | null;
  exhausted_at?: string | null;
}

interface KeySummary {
  total_keys: number;
  active_keys: number;
  inactive_keys: number;
  quota_exhausted_keys: number;
  total_calls: number;
  current_key: ApiKey | null;
}

const App: React.FC = () => {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKey, setNewKey] = useState({ service: 'ollama', key: '' });
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'keys' | 'chat' | 'logs'>('keys');
  const [copiedBase, setCopiedBase] = useState(false);
  const [summary, setSummary] = useState<KeySummary | null>(null);
  const [isTestModalOpen, setIsTestModalOpen] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [keyToTest, setKeyToTest] = useState<ApiKey | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<ApiKey | null>(null);
  const [isAddKeyModalOpen, setIsAddKeyModalOpen] = useState(false);
  const [addKeyResult, setAddKeyResult] = useState<string | null>(null);

  const baseUrl = `http://127.0.0.1:7317/v1`;

  const fetchKeys = async () => {
    try {
      const [keysResponse, summaryResponse] = await Promise.all([
        fetch('/admin/keys'),
        fetch('/admin/keys/summary'),
        ]);
      const [keysData, summaryData] = await Promise.all([
        keysResponse.json(),
        summaryResponse.json(),
        ]);
      setKeys(keysData);
      setSummary(summaryData);
    } catch (error) {
      console.error('Failed to fetch keys', error);
    }
  };

  const addKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKey.key) return;

    // Check for duplicate keys before adding
    const isDuplicate = keys.some(
      (key) => key.service === newKey.service && key.key === newKey.key
    );

    if (isDuplicate) {
      setIsAddKeyModalOpen(true);
      setAddKeyResult('Error: This API key already exists for this service.');
      return;
    }

    setLoading(true);
    setIsAddKeyModalOpen(true);
    setAddKeyResult('Adding key...');
    try {
      const response = await fetch('/admin/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newKey),
      });
      const data = await response.json();
      if (response.ok) {
        setNewKey({ ...newKey, key: '' });
        setAddKeyResult('Key added successfully!');
        fetchKeys();
      } else {
        const message = data.message || data.detail || data.error?.message || 'Unknown error';
        setAddKeyResult(`Failed to add key: ${message}`);
      }
    } catch (error) {
      console.error('Failed to add key', error);
      setAddKeyResult('Error: Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const confirmDeleteKey = (key: ApiKey) => {
    setKeyToDelete(key);
    setIsDeleteModalOpen(true);
  };

  const deleteKey = async () => {
    if (!keyToDelete) return;
    try {
      await fetch(`/admin/keys/${keyToDelete.id}`, { method: 'DELETE' });
      fetchKeys();
      setIsDeleteModalOpen(false);
      setKeyToDelete(null);
    } catch (error) {
      console.error('Failed to delete key', error);
      setIsDeleteModalOpen(false);
      setKeyToDelete(null);
    }
  };

  const openTestModal = (key: ApiKey) => {
    setKeyToTest(key);
    setIsTestModalOpen(true);
    runKeyTest(key.id!);
  };

  const runKeyTest = async (id: number) => {
    setTestResult('Testing...');
    try {
      const response = await fetch(`/admin/keys/${id}/test`, { method: 'POST' });
      const data = await response.json();
      const message = data.message || data.detail || data.error?.message || 'Unknown test result';
      setTestResult(message);
      fetchKeys();
    } catch (error) {
      console.error('Failed to test key', error);
      setTestResult('Error: Failed to connect to server');
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

  useEffect(() => {
    if (isTestModalOpen && testResult && testResult !== 'Testing...') {
      const timer = setTimeout(() => {
        setIsTestModalOpen(false);
        setTestResult(null);
        setKeyToTest(null);
      }, 6000); // Autoclose after 6 seconds
      return () => clearTimeout(timer);
    }
  }, [isTestModalOpen, testResult]);

  useEffect(() => {
    if (isAddKeyModalOpen && addKeyResult && addKeyResult !== 'Adding key...') {
      const timer = setTimeout(() => {
        setIsAddKeyModalOpen(false);
        setAddKeyResult(null);
      }, 6000); // Autoclose after 6 seconds
      return () => clearTimeout(timer);
    }
  }, [isAddKeyModalOpen, addKeyResult]);

  const groupedKeys = keys.reduce((acc, key) => {
    if (!acc[key.service]) acc[key.service] = [];
    acc[key.service].push(key);
    return acc;
  }, {} as Record<string, ApiKey[]>);

  const currentKeyId = summary?.current_key?.id;

  const formatTimestamp = (value?: string | null) => {
    if (!value) return 'Never';

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }

    return date.toLocaleString();
  };

  const formatServiceLabel = (service: string) => service.replace(/_/g, ' ');

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
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-card p-5 border-brand-500/30 bg-brand-500/5">
                <p className="text-[11px] text-slate-500 uppercase tracking-[0.24em] mb-2">Current API Used</p>
                {summary?.current_key ? (
                  <>
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <span className="text-lg font-semibold capitalize">{formatServiceLabel(summary.current_key.service)}</span>
                      <span className="text-xs px-2 py-1 rounded-full bg-brand-500/15 text-brand-300">
                        #{summary.current_key.id}
                      </span>
                    </div>
                    <p className="text-sm font-mono text-slate-200 mb-1">••••••••{summary.current_key.key.slice(-4)}</p>
                    <p className="text-sm text-slate-400">{summary.current_key.last_used_provider_id || 'Provider pending'}</p>
                    <p className="text-xs text-slate-500 mt-2">{summary.current_key.last_used_model || 'No model recorded yet'}</p>
                    <p className="text-xs text-slate-500 mt-1">Last used {formatTimestamp(summary.current_key.last_used)}</p>
                  </>
                ) : (
                  <p className="text-sm text-slate-500">No routed requests yet.</p>
                )}
              </div>

              <div className="glass-card p-5">
                <p className="text-[11px] text-slate-500 uppercase tracking-[0.24em] mb-2">Total Calls</p>
                <p className="text-3xl font-semibold text-slate-100">{summary?.total_calls ?? 0}</p>
                <p className="text-sm text-slate-500 mt-2">Successful and failed key attempts combined.</p>
              </div>

              <div className="glass-card p-5 border-rose-500/30 bg-rose-500/5">
                <p className="text-[11px] text-slate-500 uppercase tracking-[0.24em] mb-2">Inactive Keys</p>
                <p className="text-3xl font-semibold text-rose-300">{summary?.inactive_keys ?? 0}</p>
                <p className="text-sm text-slate-500 mt-2">Quota exhausted: {summary?.quota_exhausted_keys ?? 0}</p>
              </div>
            </section>

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
                  <option value="gemini">Google Gemini</option>
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
                    {formatServiceLabel(service)} Keys
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {serviceKeys
                      .slice()
                      .sort((left, right) => {
                        const currentScore = Number(right.id === currentKeyId) - Number(left.id === currentKeyId);
                        if (currentScore !== 0) return currentScore;
                        const countScore = (right.request_count || 0) - (left.request_count || 0);
                        if (countScore !== 0) return countScore;
                        return (right.last_used || '').localeCompare(left.last_used || '');
                      })
                      .map((key) => {
                        const isCurrent = key.id === currentKeyId;
                        const isQuotaExhausted = key.status === 'quota_exhausted';
                        const isInactive = key.status !== 'active';

                        return (
                      <div
                        key={key.id}
                        className={`glass-card p-5 relative group overflow-hidden ${
                          isQuotaExhausted
                            ? 'border-rose-500/40 bg-rose-500/8'
                            : isCurrent
                              ? 'border-brand-500/35 bg-brand-500/6'
                              : isInactive
                                ? 'border-amber-500/30 bg-amber-500/6'
                                : ''
                        }`}
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-center gap-2">
                            <div className="bg-white/5 p-2 rounded-lg">
                              <Key className="w-5 h-5 text-brand-300" />
                            </div>
                            <button
                              onClick={() => openTestModal(key)}
                              className="text-emerald-400 hover:text-emerald-300 p-2 rounded-lg hover:bg-emerald-400/10 transition-all"
                              title="Test Key"
                            >
                              <Activity className="w-4 h-4" />
                            </button>
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            {isCurrent && (
                              <div className="text-[10px] px-2 py-1 rounded-full bg-brand-500/20 text-brand-300 uppercase tracking-wider font-semibold">
                                Current Route
                              </div>
                            )}
                            <div className={`text-xs px-2 py-1 rounded-full flex items-center gap-1 ${
                              key.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                              key.status === 'quota_exhausted' ? 'bg-rose-500/20 text-rose-300' :
                              key.status === 'rate_limited' ? 'bg-amber-500/20 text-amber-400' :
                              'bg-rose-500/20 text-rose-400'
                            }`}>
                              {key.status === 'active' ? <ShieldCheck className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
                              {key.status.replace('_', ' ')}
                            </div>
                          </div>
                        </div>
                        <div className="mb-4 flex items-center justify-between gap-3 rounded-xl bg-black/20 px-3 py-2 border border-white/5">
                          <div>
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Call Count</p>
                            <p className="text-2xl font-semibold text-slate-100">{key.request_count || 0}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Last Used</p>
                            <p className="text-xs text-slate-300">{formatTimestamp(key.last_used)}</p>
                          </div>
                        </div>
                        <div className="mb-6">
                          <p className="text-[10px] text-slate-500 mb-1 font-bold uppercase tracking-widest">Key Suffix</p>
                          <code className="text-sm font-mono text-slate-300">
                            ••••••••{key.key.slice(-4)}
                          </code>
                        </div>
                        <div className="space-y-2 text-sm text-slate-400 mb-8">
                          <p>
                            <span className="text-slate-500">Provider:</span> {key.last_used_provider_id || 'Unused'}
                          </p>
                          <p>
                            <span className="text-slate-500">Model:</span> {key.last_used_model || 'Unused'}
                          </p>
                        </div>
                        {key.last_status_message && (
                          <div className={`mb-8 rounded-xl border px-3 py-3 text-sm ${
                            isInactive ? 'border-rose-500/25 bg-rose-500/10 text-rose-200' : 'border-white/10 bg-white/5 text-slate-300'
                          }`}>
                            <p className="text-[10px] uppercase tracking-widest font-semibold mb-1 text-slate-400">Last Provider Message</p>
                            <p>{key.last_status_message}</p>
                            {key.exhausted_at && (
                              <p className="mt-2 text-xs text-rose-200/80">Exhausted at {formatTimestamp(key.exhausted_at)}</p>
                            )}
                          </div>
                        )}
                        <button
                          onClick={() => confirmDeleteKey(key)}
                          className="absolute bottom-4 right-4 text-slate-500 hover:text-rose-400 p-2 rounded-lg hover:bg-rose-400/10 transition-all opacity-0 group-hover:opacity-100"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                        );
                      })}
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

        <Modal isOpen={isTestModalOpen} onClose={() => setIsTestModalOpen(false)} title="Test API Key">
          {keyToTest && (
            <div className="space-y-4">
              <p>Testing key for service: <span className="font-semibold capitalize">{formatServiceLabel(keyToTest.service)}</span></p>
              <p>Key Suffix: <code className="font-mono">••••••••{keyToTest.key.slice(-4)}</code></p>
              <p className="font-semibold">Result:</p>
              <div className="bg-slate-700 p-3 rounded-md font-mono text-sm overflow-x-auto">
                {testResult}
              </div>
            </div>
          )}
          <div className="flex justify-end space-x-2 mt-4">
            <button onClick={() => setIsTestModalOpen(false)} className="btn-secondary">Close</button>
          </div>
        </Modal>

        <Modal isOpen={isDeleteModalOpen} onClose={() => setIsDeleteModalOpen(false)} title="Confirm Deletion">
          {keyToDelete && (
            <div className="space-y-4">
              <p>Are you sure you want to delete the API key for <span className="font-semibold capitalize">{formatServiceLabel(keyToDelete.service)}</span> with suffix <code className="font-mono">••••••••{keyToDelete.key.slice(-4)}</code>?</p>
              <p className="text-sm text-rose-300">This action cannot be undone.</p>
            </div>
          )}
          <div className="flex justify-end space-x-2 mt-4">
            <button onClick={() => setIsDeleteModalOpen(false)} className="btn-secondary">Cancel</button>
            <button onClick={deleteKey} className="btn-danger">Delete</button>
          </div>
        </Modal>

        <Modal isOpen={isAddKeyModalOpen} onClose={() => setIsAddKeyModalOpen(false)} title="Add API Key Result">
          <div className="space-y-4">
            <p className="font-semibold">Result:</p>
            <div className="bg-slate-700 p-3 rounded-md font-mono text-sm overflow-x-auto">
              {addKeyResult}
            </div>
          </div>
          <div className="flex justify-end space-x-2 mt-4">
            <button onClick={() => setIsAddKeyModalOpen(false)} className="btn-secondary">Close</button>
          </div>
        </Modal>
      </main>
    </div>
  );
};

export default App;
