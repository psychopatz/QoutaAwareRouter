import React, { useState, useEffect } from 'react';
import { Activity, Server, AlertTriangle } from 'lucide-react';

interface TrafficLog {
  id: string;
  timestamp: number;
  method: string;
  path: string;
  model: string;
  provider_id?: string;
  key_id?: number;
  key_suffix?: string;
  status_code: number;
  latency_ms: number;
  error?: string;
}

const TrafficLogs: React.FC = () => {
  const [logs, setLogs] = useState<TrafficLog[]>([]);

  const fetchLogs = async () => {
    try {
      const response = await fetch('/admin/traffic');
      const data = await response.json();
      setLogs(data);
    } catch (error) {
      console.error('Failed to fetch logs', error);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-card overflow-hidden">
      <div className="p-4 border-b border-white/10 bg-white/5 flex items-center gap-2">
        <Activity className="w-5 h-5 text-amber-400" />
        <span className="font-medium">Live Traffic Stream</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-white/5 text-slate-400 uppercase text-xs">
            <tr>
              <th className="px-4 py-3 font-medium">Time</th>
              <th className="px-4 py-3 font-medium">Model</th>
              <th className="px-4 py-3 font-medium">Provider</th>
              <th className="px-4 py-3 font-medium">API Key</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium text-right">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {logs.map((log) => (
              <tr key={log.id} className="hover:bg-white/5 transition-colors group">
                <td className="px-4 py-3 text-slate-500 whitespace-nowrap">
                  {new Date(log.timestamp * 1000).toLocaleTimeString()}
                </td>
                <td className="px-4 py-3 font-mono">{log.model}</td>
                <td className="px-4 py-3">
                  {log.provider_id ? (
                    <span className="flex items-center gap-1">
                      <Server className="w-3 h-3 text-brand-400" />
                      {log.provider_id}
                    </span>
                  ) : '-'}
                </td>
                <td className="px-4 py-3 font-mono text-xs text-slate-300">
                  {log.key_id ? `#${log.key_id} ••••${log.key_suffix || '????'}` : '-'}
                </td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded-full text-xs flex items-center gap-1 w-fit ${
                    log.status_code < 300 ? 'bg-emerald-500/20 text-emerald-400' : 
                    'bg-rose-500/20 text-rose-400'
                  }`}>
                    {log.status_code}
                    {log.error && <AlertTriangle className="w-3 h-3" />}
                  </span>
                </td>
                <td className="px-4 py-3 text-right text-slate-400">
                  {log.latency_ms.toFixed(0)}ms
                </td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-20 text-center text-slate-500">
                  No traffic recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TrafficLogs;
