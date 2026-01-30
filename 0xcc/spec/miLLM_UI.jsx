import React, { useState, useEffect } from 'react';
import { 
  Cpu, HardDrive, Zap, Thermometer, Download, Play, Square, Trash2, 
  Settings, Link2, Unlink, Save, Upload, Plus, Search, ChevronDown,
  Check, AlertCircle, Activity, Eye, EyeOff, RefreshCw, Server,
  Database, Sliders, Layers, FileJson, Copy, MoreVertical, X
} from 'lucide-react';

// Mock data
const mockModels = [
  { id: 1, name: 'gemma-2-2b', repo: 'google/gemma-2-2b', params: '2.5B', quantization: 'Q4', memory: '1.8 GB', status: 'loaded' },
  { id: 2, name: 'gemma-2-9b', repo: 'google/gemma-2-9b', params: '9B', quantization: 'Q4', memory: '5.2 GB', status: 'ready' },
  { id: 3, name: 'llama-3.2-3b', repo: 'meta-llama/Llama-3.2-3B', params: '3B', quantization: 'FP16', memory: '6.4 GB', status: 'ready' },
];

const mockSAEs = [
  { id: 1, name: 'gemma-scope-2b-L12', repo: 'google/gemma-scope-2b-pt-res', layer: 12, features: 16384, size: '256 MB', status: 'attached', linkedModel: 'gemma-2-2b' },
  { id: 2, name: 'gemma-scope-2b-L6', repo: 'google/gemma-scope-2b-pt-res', layer: 6, features: 16384, size: '256 MB', status: 'ready', linkedModel: null },
  { id: 3, name: 'gemma-scope-9b-L20', repo: 'google/gemma-scope-9b-pt-res', layer: 20, features: 32768, size: '512 MB', status: 'ready', linkedModel: null },
];

const mockProfiles = [
  { id: 1, name: 'yelling-demo', model: 'gemma-2-2b', sae: 'gemma-scope-2b-L12', features: [{ index: 1234, strength: 5.0, label: 'Yelling/Capitalization' }], active: true },
  { id: 2, name: 'formal-tone', model: 'gemma-2-2b', sae: 'gemma-scope-2b-L12', features: [{ index: 892, strength: 3.0, label: 'Formal Language' }, { index: 1456, strength: -2.0, label: 'Casual Speech' }], active: false },
  { id: 3, name: 'creative-writing', model: 'gemma-2-2b', sae: 'gemma-scope-2b-L12', features: [{ index: 2341, strength: 4.0, label: 'Metaphorical Language' }, { index: 3120, strength: 2.5, label: 'Descriptive Detail' }], active: false },
];

const mockFeatures = [
  { index: 1234, label: 'Yelling/Capitalization', activation: 0.82 },
  { index: 892, label: 'Formal Language', activation: 0.45 },
  { index: 1456, label: 'Casual Speech', activation: 0.23 },
  { index: 2341, label: 'Metaphorical Language', activation: 0.67 },
  { index: 3120, label: 'Descriptive Detail', activation: 0.51 },
  { index: 4521, label: 'Technical Jargon', activation: 0.12 },
  { index: 5678, label: 'Emotional Tone', activation: 0.38 },
  { index: 6789, label: 'Question Asking', activation: 0.71 },
];

const mockMonitorData = [
  { time: '12:00:01', features: { 1234: 0.82, 892: 0.45, 2341: 0.67 } },
  { time: '12:00:02', features: { 1234: 0.91, 892: 0.52, 2341: 0.58 } },
  { time: '12:00:03', features: { 1234: 0.78, 892: 0.48, 2341: 0.72 } },
  { time: '12:00:04', features: { 1234: 0.85, 892: 0.41, 2341: 0.65 } },
  { time: '12:00:05', features: { 1234: 0.95, 892: 0.55, 2341: 0.61 } },
];

// Styles
const styles = {
  app: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #0a0f1a 0%, #0f172a 50%, #0a1628 100%)',
    color: '#e2e8f0',
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
  },
  header: {
    background: 'rgba(15, 23, 42, 0.95)',
    borderBottom: '1px solid rgba(56, 189, 248, 0.1)',
    padding: '0 24px',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    backdropFilter: 'blur(12px)',
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: '64px',
    maxWidth: '1600px',
    margin: '0 auto',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  logoIcon: {
    width: '36px',
    height: '36px',
    background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: {
    fontSize: '20px',
    fontWeight: '700',
    color: '#f1f5f9',
    letterSpacing: '-0.5px',
  },
  logoSubtext: {
    fontSize: '11px',
    color: '#64748b',
    marginTop: '2px',
  },
  nav: {
    display: 'flex',
    gap: '4px',
  },
  navItem: {
    padding: '8px 16px',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: '500',
    color: '#94a3b8',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    border: 'none',
    background: 'transparent',
  },
  navItemActive: {
    color: '#22d3ee',
    background: 'rgba(34, 211, 238, 0.1)',
  },
  statusBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    fontSize: '12px',
  },
  statusItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#64748b',
  },
  statusValue: {
    color: '#e2e8f0',
    fontWeight: '600',
    fontFamily: "'JetBrains Mono', monospace",
  },
  main: {
    maxWidth: '1600px',
    margin: '0 auto',
    padding: '24px',
  },
  pageHeader: {
    marginBottom: '24px',
  },
  pageTitle: {
    fontSize: '24px',
    fontWeight: '700',
    color: '#f1f5f9',
    marginBottom: '4px',
  },
  pageSubtitle: {
    fontSize: '14px',
    color: '#64748b',
  },
  card: {
    background: 'rgba(15, 23, 42, 0.6)',
    border: '1px solid rgba(56, 189, 248, 0.1)',
    borderRadius: '12px',
    padding: '20px',
    marginBottom: '20px',
  },
  cardTitle: {
    fontSize: '14px',
    fontWeight: '600',
    color: '#f1f5f9',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  inputGroup: {
    marginBottom: '16px',
  },
  label: {
    display: 'block',
    fontSize: '12px',
    fontWeight: '500',
    color: '#94a3b8',
    marginBottom: '6px',
  },
  input: {
    width: '100%',
    padding: '10px 14px',
    background: 'rgba(30, 41, 59, 0.5)',
    border: '1px solid rgba(71, 85, 105, 0.5)',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontSize: '14px',
    outline: 'none',
    transition: 'border-color 0.2s ease',
    boxSizing: 'border-box',
  },
  select: {
    width: '100%',
    padding: '10px 14px',
    background: 'rgba(30, 41, 59, 0.5)',
    border: '1px solid rgba(71, 85, 105, 0.5)',
    borderRadius: '8px',
    color: '#e2e8f0',
    fontSize: '14px',
    outline: 'none',
    cursor: 'pointer',
    appearance: 'none',
  },
  buttonRow: {
    display: 'flex',
    gap: '12px',
  },
  buttonPrimary: {
    flex: 1,
    padding: '12px 20px',
    background: 'linear-gradient(135deg, #0891b2 0%, #06b6d4 100%)',
    border: 'none',
    borderRadius: '8px',
    color: '#fff',
    fontSize: '14px',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    transition: 'all 0.2s ease',
  },
  buttonSecondary: {
    flex: 1,
    padding: '12px 20px',
    background: 'transparent',
    border: '1px solid rgba(71, 85, 105, 0.5)',
    borderRadius: '8px',
    color: '#94a3b8',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    transition: 'all 0.2s ease',
  },
  listItem: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px',
    background: 'rgba(30, 41, 59, 0.3)',
    border: '1px solid rgba(71, 85, 105, 0.3)',
    borderRadius: '10px',
    marginBottom: '12px',
    transition: 'all 0.2s ease',
  },
  listItemInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
  },
  listItemIcon: {
    width: '40px',
    height: '40px',
    background: 'rgba(34, 211, 238, 0.1)',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#22d3ee',
  },
  listItemName: {
    fontSize: '15px',
    fontWeight: '600',
    color: '#f1f5f9',
  },
  listItemMeta: {
    fontSize: '12px',
    color: '#64748b',
    marginTop: '2px',
  },
  listItemActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  badge: {
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  badgeLoaded: {
    background: 'rgba(34, 197, 94, 0.15)',
    color: '#4ade80',
  },
  badgeReady: {
    background: 'rgba(34, 211, 238, 0.15)',
    color: '#22d3ee',
  },
  badgeAttached: {
    background: 'rgba(168, 85, 247, 0.15)',
    color: '#c084fc',
  },
  badgeActive: {
    background: 'rgba(251, 191, 36, 0.15)',
    color: '#fbbf24',
  },
  actionButton: {
    padding: '8px 14px',
    background: 'rgba(34, 211, 238, 0.1)',
    border: '1px solid rgba(34, 211, 238, 0.3)',
    borderRadius: '6px',
    color: '#22d3ee',
    fontSize: '12px',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    transition: 'all 0.2s ease',
  },
  iconButton: {
    padding: '8px',
    background: 'transparent',
    border: 'none',
    borderRadius: '6px',
    color: '#64748b',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
  },
  slider: {
    width: '100%',
    height: '6px',
    borderRadius: '3px',
    background: 'rgba(71, 85, 105, 0.5)',
    appearance: 'none',
    cursor: 'pointer',
  },
  featureCard: {
    background: 'rgba(30, 41, 59, 0.5)',
    border: '1px solid rgba(71, 85, 105, 0.3)',
    borderRadius: '10px',
    padding: '16px',
    marginBottom: '12px',
  },
  featureHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '12px',
  },
  featureIndex: {
    fontSize: '13px',
    fontWeight: '700',
    color: '#22d3ee',
    fontFamily: "'JetBrains Mono', monospace",
  },
  featureLabel: {
    fontSize: '13px',
    color: '#94a3b8',
    marginLeft: '8px',
  },
  strengthDisplay: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  strengthValue: {
    fontSize: '14px',
    fontWeight: '700',
    color: '#f1f5f9',
    fontFamily: "'JetBrains Mono', monospace",
    minWidth: '50px',
    textAlign: 'right',
  },
  activationBar: {
    height: '4px',
    background: 'rgba(71, 85, 105, 0.5)',
    borderRadius: '2px',
    overflow: 'hidden',
    marginTop: '8px',
  },
  activationFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #06b6d4 0%, #22d3ee 100%)',
    borderRadius: '2px',
    transition: 'width 0.3s ease',
  },
  monitorGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '12px',
    marginBottom: '20px',
  },
  monitorCard: {
    background: 'rgba(30, 41, 59, 0.5)',
    border: '1px solid rgba(71, 85, 105, 0.3)',
    borderRadius: '10px',
    padding: '16px',
    textAlign: 'center',
  },
  monitorValue: {
    fontSize: '28px',
    fontWeight: '700',
    color: '#f1f5f9',
    fontFamily: "'JetBrains Mono', monospace",
  },
  monitorLabel: {
    fontSize: '11px',
    color: '#64748b',
    marginTop: '4px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  tableHeader: {
    background: 'rgba(30, 41, 59, 0.5)',
  },
  tableHeaderCell: {
    padding: '12px 16px',
    textAlign: 'left',
    fontSize: '11px',
    fontWeight: '600',
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    borderBottom: '1px solid rgba(71, 85, 105, 0.3)',
  },
  tableCell: {
    padding: '14px 16px',
    fontSize: '13px',
    color: '#e2e8f0',
    borderBottom: '1px solid rgba(71, 85, 105, 0.2)',
  },
  emptyState: {
    textAlign: 'center',
    padding: '40px 20px',
    color: '#64748b',
  },
  twoColumn: {
    display: 'grid',
    gridTemplateColumns: '340px 1fr',
    gap: '24px',
  },
  serverStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    background: 'rgba(34, 197, 94, 0.1)',
    border: '1px solid rgba(34, 197, 94, 0.3)',
    borderRadius: '8px',
    marginLeft: '16px',
  },
  serverStatusDot: {
    width: '8px',
    height: '8px',
    background: '#4ade80',
    borderRadius: '50%',
    animation: 'pulse 2s infinite',
  },
  serverStatusText: {
    fontSize: '12px',
    color: '#4ade80',
    fontWeight: '600',
  },
  profileInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 12px',
    background: 'rgba(251, 191, 36, 0.1)',
    border: '1px solid rgba(251, 191, 36, 0.3)',
    borderRadius: '6px',
    marginLeft: '8px',
  },
  profileInfoText: {
    fontSize: '12px',
    color: '#fbbf24',
    fontWeight: '500',
  },
};

// Components
const StatusBar = ({ model, sae, profile }) => (
  <div style={styles.statusBar}>
    <div style={styles.statusItem}>
      <Cpu size={14} />
      <span>CPU:</span>
      <span style={styles.statusValue}>12%</span>
    </div>
    <div style={styles.statusItem}>
      <HardDrive size={14} />
      <span>RAM:</span>
      <span style={styles.statusValue}>8.2/32 GB</span>
    </div>
    <div style={styles.statusItem}>
      <Zap size={14} />
      <span>GPU:</span>
      <span style={styles.statusValue}>45%</span>
    </div>
    <div style={styles.statusItem}>
      <Database size={14} />
      <span>VRAM:</span>
      <span style={styles.statusValue}>4.2/24 GB</span>
    </div>
    <div style={styles.statusItem}>
      <Thermometer size={14} />
      <span style={styles.statusValue}>52°C</span>
    </div>
    <div style={styles.serverStatus}>
      <div style={styles.serverStatusDot}></div>
      <span style={styles.serverStatusText}>Serving</span>
    </div>
    {model && (
      <div style={{ ...styles.statusItem, marginLeft: '8px' }}>
        <Server size={14} style={{ color: '#4ade80' }} />
        <span style={{ color: '#4ade80' }}>{model}</span>
      </div>
    )}
    {sae && (
      <div style={styles.statusItem}>
        <Layers size={14} style={{ color: '#c084fc' }} />
        <span style={{ color: '#c084fc' }}>{sae}</span>
      </div>
    )}
    {profile && (
      <div style={styles.profileInfo}>
        <Sliders size={12} />
        <span style={styles.profileInfoText}>{profile}</span>
      </div>
    )}
  </div>
);

const ModelsPage = ({ models, onLoadModel }) => {
  const [repoId, setRepoId] = useState('');
  const [quantization, setQuantization] = useState('Q4');
  
  return (
    <div>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>Models</h1>
        <p style={styles.pageSubtitle}>Download and manage PyTorch models with quantization support</p>
      </div>
      
      <div style={styles.card}>
        <div style={styles.cardTitle}>
          <Download size={16} style={{ color: '#22d3ee' }} />
          Download from HuggingFace
        </div>
        <div style={styles.grid}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>HuggingFace Model Repository</label>
            <input 
              style={styles.input} 
              placeholder="e.g., google/gemma-2-2b"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
            />
          </div>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Quantization Format</label>
            <select style={styles.select} value={quantization} onChange={(e) => setQuantization(e.target.value)}>
              <option value="Q4">Q4 (4-bit) - Recommended</option>
              <option value="Q8">Q8 (8-bit)</option>
              <option value="FP16">FP16 (Full Precision)</option>
            </select>
          </div>
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Access Token <span style={{ color: '#64748b' }}>(optional, for gated models)</span></label>
          <input style={styles.input} placeholder="hf_xxxxxxxxxxxxxxxxxxxx" type="password" />
        </div>
        <div style={{ ...styles.inputGroup, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input type="checkbox" id="trustRemote" style={{ accentColor: '#22d3ee' }} />
          <label htmlFor="trustRemote" style={{ fontSize: '13px', color: '#fbbf24' }}>
            Trust Remote Code <span style={{ color: '#64748b', fontWeight: 'normal' }}>- Required for some models (Phi-4, CodeLlama, etc.)</span>
          </label>
        </div>
        <div style={styles.buttonRow}>
          <button style={styles.buttonSecondary}>
            <Eye size={16} /> Preview
          </button>
          <button style={styles.buttonPrimary}>
            <Download size={16} /> Download
          </button>
        </div>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: '600', color: '#f1f5f9' }}>Your Models ({models.length})</h2>
      </div>
      
      {models.map((model) => (
        <div key={model.id} style={styles.listItem}>
          <div style={styles.listItemInfo}>
            <div style={styles.listItemIcon}>
              <Server size={20} />
            </div>
            <div>
              <div style={styles.listItemName}>{model.name}</div>
              <div style={styles.listItemMeta}>
                {model.params} params • {model.quantization} quantization • {model.memory} memory
              </div>
              <div style={{ ...styles.listItemMeta, fontFamily: "'JetBrains Mono', monospace", fontSize: '11px' }}>
                {model.repo}
              </div>
            </div>
          </div>
          <div style={styles.listItemActions}>
            {model.status === 'loaded' ? (
              <>
                <span style={{ ...styles.badge, ...styles.badgeLoaded }}>Loaded</span>
                <button style={{ ...styles.actionButton, background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.3)', color: '#f87171' }}>
                  <Square size={14} /> Unload
                </button>
              </>
            ) : (
              <>
                <span style={{ ...styles.badge, ...styles.badgeReady }}>Ready</span>
                <button style={styles.actionButton} onClick={() => onLoadModel(model.name)}>
                  <Play size={14} /> Load
                </button>
              </>
            )}
            <button style={styles.iconButton}><Trash2 size={16} /></button>
          </div>
        </div>
      ))}
    </div>
  );
};

const SAEsPage = ({ saes, models, onAttachSAE }) => {
  const [repoId, setRepoId] = useState('');
  const [linkedModel, setLinkedModel] = useState('');
  
  return (
    <div>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>Sparse Autoencoders (SAEs)</h1>
        <p style={styles.pageSubtitle}>Download and manage SAEs for feature steering. Supports SAELens format from HuggingFace.</p>
      </div>
      
      <div style={styles.card}>
        <div style={styles.cardTitle}>
          <Download size={16} style={{ color: '#22d3ee' }} />
          Download from HuggingFace
        </div>
        <div style={styles.grid}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>HuggingFace Repository</label>
            <input 
              style={styles.input} 
              placeholder="e.g., google/gemma-scope-2b-pt-res"
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
            />
          </div>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Custom Name <span style={{ color: '#64748b' }}>(optional)</span></label>
            <input style={styles.input} placeholder="Auto-generated from file path" />
          </div>
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Access Token <span style={{ color: '#64748b' }}>(optional, for gated repos)</span></label>
          <input style={styles.input} placeholder="hf_xxxxxxxxxxxxxxxxxxxx" type="password" />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Link to Model <span style={{ color: '#f87171' }}>*</span></label>
          <select style={styles.select} value={linkedModel} onChange={(e) => setLinkedModel(e.target.value)}>
            <option value="">Select a model...</option>
            {models.map(m => (
              <option key={m.id} value={m.name}>{m.name}</option>
            ))}
          </select>
          <p style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>Select a downloaded model to use with this SAE for steering</p>
        </div>
        <button style={{ ...styles.buttonSecondary, width: '100%' }}>
          <Search size={16} /> Preview Repository
        </button>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: '600', color: '#f1f5f9' }}>Your SAEs ({saes.length})</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input 
            style={{ ...styles.input, width: '240px', padding: '8px 12px' }} 
            placeholder="Search SAEs by name, model, or repo..."
          />
        </div>
      </div>
      
      {saes.map((sae) => (
        <div key={sae.id} style={styles.listItem}>
          <div style={styles.listItemInfo}>
            <div style={{ ...styles.listItemIcon, background: 'rgba(168, 85, 247, 0.1)', color: '#c084fc' }}>
              <Layers size={20} />
            </div>
            <div>
              <div style={styles.listItemName}>{sae.name}</div>
              <div style={styles.listItemMeta}>
                {sae.features.toLocaleString()} features • Layer {sae.layer} • {sae.size}
              </div>
              <div style={{ ...styles.listItemMeta, fontFamily: "'JetBrains Mono', monospace", fontSize: '11px' }}>
                {sae.repo}
              </div>
            </div>
          </div>
          <div style={styles.listItemActions}>
            {sae.linkedModel && (
              <span style={{ fontSize: '12px', color: '#64748b', marginRight: '8px' }}>
                <Link2 size={12} style={{ display: 'inline', marginRight: '4px' }} />
                {sae.linkedModel}
              </span>
            )}
            {sae.status === 'attached' ? (
              <>
                <span style={{ ...styles.badge, ...styles.badgeAttached }}>Attached</span>
                <button style={{ ...styles.actionButton, background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.3)', color: '#f87171' }}>
                  <Unlink size={14} /> Detach
                </button>
              </>
            ) : (
              <>
                <span style={{ ...styles.badge, ...styles.badgeReady }}>Ready</span>
                <button style={styles.actionButton} onClick={() => onAttachSAE(sae.name)}>
                  <Link2 size={14} /> Attach
                </button>
              </>
            )}
            <button style={styles.iconButton}><Trash2 size={16} /></button>
          </div>
        </div>
      ))}
    </div>
  );
};

const SteeringPage = ({ saes, features, onUpdateFeature }) => {
  const [selectedSAE, setSelectedSAE] = useState('gemma-scope-2b-L12');
  const [steeringActive, setSteeringActive] = useState(true);
  const [selectedFeatures, setSelectedFeatures] = useState([
    { index: 1234, strength: 5.0, label: 'Yelling/Capitalization' },
    { index: 892, strength: 3.0, label: 'Formal Language' },
  ]);
  const [searchQuery, setSearchQuery] = useState('');
  
  const addFeature = (feature) => {
    if (!selectedFeatures.find(f => f.index === feature.index)) {
      setSelectedFeatures([...selectedFeatures, { ...feature, strength: 0 }]);
    }
  };
  
  const removeFeature = (index) => {
    setSelectedFeatures(selectedFeatures.filter(f => f.index !== index));
  };
  
  const updateStrength = (index, strength) => {
    setSelectedFeatures(selectedFeatures.map(f => 
      f.index === index ? { ...f, strength: parseFloat(strength) } : f
    ));
  };
  
  const filteredFeatures = features.filter(f => 
    !selectedFeatures.find(sf => sf.index === f.index) &&
    (f.label.toLowerCase().includes(searchQuery.toLowerCase()) || 
     f.index.toString().includes(searchQuery))
  );
  
  return (
    <div style={styles.twoColumn}>
      <div>
        <div style={styles.card}>
          <div style={styles.cardTitle}>
            <Settings size={16} style={{ color: '#22d3ee' }} />
            Select SAE
          </div>
          <select 
            style={styles.select} 
            value={selectedSAE} 
            onChange={(e) => setSelectedSAE(e.target.value)}
          >
            {saes.filter(s => s.status === 'attached').map(sae => (
              <option key={sae.id} value={sae.name}>{sae.name} (L{sae.layer})</option>
            ))}
          </select>
        </div>
        
        <div style={styles.card}>
          <div style={{ ...styles.cardTitle, justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={16} style={{ color: '#22d3ee' }} />
              Selected Features ({selectedFeatures.length})
            </span>
            <button 
              style={{ ...styles.iconButton, color: '#64748b', fontSize: '12px' }}
              onClick={() => setSelectedFeatures([])}
            >
              Clear all
            </button>
          </div>
          
          {selectedFeatures.map((feature) => (
            <div key={feature.index} style={styles.featureCard}>
              <div style={styles.featureHeader}>
                <div>
                  <span style={styles.featureIndex}>#{feature.index}</span>
                  <span style={styles.featureLabel}>{feature.label}</span>
                </div>
                <button style={styles.iconButton} onClick={() => removeFeature(feature.index)}>
                  <X size={14} />
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <input
                  type="range"
                  min="-10"
                  max="10"
                  step="0.5"
                  value={feature.strength}
                  onChange={(e) => updateStrength(feature.index, e.target.value)}
                  style={{ ...styles.slider, flex: 1 }}
                />
                <span style={styles.strengthValue}>
                  {feature.strength > 0 ? '+' : ''}{feature.strength.toFixed(1)}
                </span>
              </div>
              <div style={{ 
                height: '4px', 
                background: 'rgba(71, 85, 105, 0.5)', 
                borderRadius: '2px', 
                marginTop: '8px',
                overflow: 'hidden'
              }}>
                <div style={{
                  height: '100%',
                  width: `${Math.abs(feature.strength) * 10}%`,
                  marginLeft: feature.strength < 0 ? `${50 - Math.abs(feature.strength) * 5}%` : '50%',
                  background: feature.strength >= 0 
                    ? 'linear-gradient(90deg, #06b6d4, #22d3ee)' 
                    : 'linear-gradient(90deg, #f87171, #ef4444)',
                  borderRadius: '2px',
                  transition: 'all 0.2s ease',
                }}></div>
              </div>
            </div>
          ))}
          
          {selectedFeatures.length === 0 && (
            <div style={styles.emptyState}>
              <p>No features selected. Browse features below to add.</p>
            </div>
          )}
        </div>
        
        <div style={styles.card}>
          <div style={styles.cardTitle}>
            <Search size={16} style={{ color: '#22d3ee' }} />
            Browse Features
          </div>
          <input 
            style={{ ...styles.input, marginBottom: '12px' }}
            placeholder="Search by label or feature ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {filteredFeatures.map((feature) => (
              <div 
                key={feature.index}
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  borderRadius: '6px',
                  marginBottom: '4px',
                  cursor: 'pointer',
                  transition: 'background 0.2s ease',
                  background: 'rgba(30, 41, 59, 0.3)',
                }}
                onClick={() => addFeature(feature)}
              >
                <div>
                  <span style={{ ...styles.featureIndex, fontSize: '12px' }}>#{feature.index}</span>
                  <span style={{ ...styles.featureLabel, fontSize: '12px' }}>{feature.label}</span>
                </div>
                <Plus size={14} style={{ color: '#22d3ee' }} />
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div>
        <div style={styles.card}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#f1f5f9', marginBottom: '4px' }}>Feature Steering</h2>
              <p style={{ fontSize: '13px', color: '#64748b' }}>Steer model outputs by adjusting feature activations during generation</p>
            </div>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                style={steeringActive ? { 
                  ...styles.actionButton, 
                  background: 'rgba(239, 68, 68, 0.1)', 
                  borderColor: 'rgba(239, 68, 68, 0.3)', 
                  color: '#f87171' 
                } : styles.buttonPrimary}
                onClick={() => setSteeringActive(!steeringActive)}
              >
                {steeringActive ? <><Square size={14} /> Stop Steering</> : <><Play size={14} /> Start Steering</>}
              </button>
              <button style={styles.buttonSecondary}>
                <RefreshCw size={14} /> Reset
              </button>
            </div>
          </div>
          
          <div style={{ 
            padding: '16px', 
            background: steeringActive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(251, 191, 36, 0.1)', 
            border: `1px solid ${steeringActive ? 'rgba(34, 197, 94, 0.3)' : 'rgba(251, 191, 36, 0.3)'}`,
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '20px'
          }}>
            <div style={{ 
              width: '10px', 
              height: '10px', 
              borderRadius: '50%', 
              background: steeringActive ? '#4ade80' : '#fbbf24',
              boxShadow: steeringActive ? '0 0 12px rgba(74, 222, 128, 0.5)' : 'none',
            }}></div>
            <span style={{ 
              fontSize: '14px', 
              fontWeight: '600', 
              color: steeringActive ? '#4ade80' : '#fbbf24' 
            }}>
              {steeringActive ? 'Steering Mode is ACTIVE' : 'Steering Mode is OFF'} 
              <span style={{ fontWeight: '400', color: '#94a3b8', marginLeft: '8px' }}>
                — {steeringActive ? `${selectedFeatures.length} features being applied` : 'Click "Start Steering" to begin'}
              </span>
            </span>
          </div>
          
          <div style={styles.cardTitle}>
            <Activity size={16} style={{ color: '#22d3ee' }} />
            Live Activations
          </div>
          
          <table style={styles.table}>
            <thead style={styles.tableHeader}>
              <tr>
                <th style={styles.tableHeaderCell}>Feature</th>
                <th style={styles.tableHeaderCell}>Label</th>
                <th style={styles.tableHeaderCell}>Steering</th>
                <th style={styles.tableHeaderCell}>Current Activation</th>
              </tr>
            </thead>
            <tbody>
              {selectedFeatures.map((feature) => {
                const liveFeature = features.find(f => f.index === feature.index);
                return (
                  <tr key={feature.index}>
                    <td style={{ ...styles.tableCell, fontFamily: "'JetBrains Mono', monospace", color: '#22d3ee' }}>
                      #{feature.index}
                    </td>
                    <td style={styles.tableCell}>{feature.label}</td>
                    <td style={{ ...styles.tableCell, fontFamily: "'JetBrains Mono', monospace", fontWeight: '600' }}>
                      <span style={{ color: feature.strength >= 0 ? '#4ade80' : '#f87171' }}>
                        {feature.strength > 0 ? '+' : ''}{feature.strength.toFixed(1)}
                      </span>
                    </td>
                    <td style={styles.tableCell}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ flex: 1, ...styles.activationBar }}>
                          <div style={{ 
                            ...styles.activationFill, 
                            width: `${(liveFeature?.activation || 0) * 100}%` 
                          }}></div>
                        </div>
                        <span style={{ 
                          fontFamily: "'JetBrains Mono', monospace", 
                          fontSize: '12px',
                          minWidth: '40px'
                        }}>
                          {(liveFeature?.activation || 0).toFixed(2)}
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          
          {selectedFeatures.length === 0 && (
            <div style={styles.emptyState}>
              <p>Select features from the left panel to begin steering</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ProfilesPage = ({ profiles, onActivateProfile }) => {
  const [editingProfile, setEditingProfile] = useState(null);
  
  return (
    <div>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>Steering Profiles</h1>
        <p style={styles.pageSubtitle}>Save and manage steering configurations for quick switching</p>
      </div>
      
      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px' }}>
        <button style={styles.buttonPrimary}>
          <Plus size={16} /> New Profile
        </button>
        <button style={styles.buttonSecondary}>
          <Upload size={16} /> Import from miStudio
        </button>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2 style={{ fontSize: '16px', fontWeight: '600', color: '#f1f5f9' }}>Saved Profiles ({profiles.length})</h2>
      </div>
      
      {profiles.map((profile) => (
        <div key={profile.id} style={{ ...styles.listItem, border: profile.active ? '1px solid rgba(251, 191, 36, 0.5)' : styles.listItem.border }}>
          <div style={styles.listItemInfo}>
            <div style={{ ...styles.listItemIcon, background: profile.active ? 'rgba(251, 191, 36, 0.1)' : 'rgba(34, 211, 238, 0.1)', color: profile.active ? '#fbbf24' : '#22d3ee' }}>
              <FileJson size={20} />
            </div>
            <div>
              <div style={styles.listItemName}>{profile.name}</div>
              <div style={styles.listItemMeta}>
                Model: {profile.model} • SAE: {profile.sae} • {profile.features.length} feature{profile.features.length !== 1 ? 's' : ''}
              </div>
              <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {profile.features.map((f, idx) => (
                  <span key={idx} style={{ 
                    fontSize: '11px', 
                    padding: '2px 8px', 
                    background: 'rgba(30, 41, 59, 0.8)', 
                    borderRadius: '4px',
                    color: '#94a3b8',
                    fontFamily: "'JetBrains Mono', monospace"
                  }}>
                    #{f.index}: {f.strength > 0 ? '+' : ''}{f.strength}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div style={styles.listItemActions}>
            {profile.active ? (
              <span style={{ ...styles.badge, ...styles.badgeActive }}>Active</span>
            ) : (
              <button style={styles.actionButton} onClick={() => onActivateProfile(profile.id)}>
                <Play size={14} /> Activate
              </button>
            )}
            <button style={styles.iconButton}><Copy size={16} /></button>
            <button style={styles.iconButton}><Settings size={16} /></button>
            <button style={styles.iconButton}><Trash2 size={16} /></button>
          </div>
        </div>
      ))}
      
      <div style={{ ...styles.card, marginTop: '24px' }}>
        <div style={styles.cardTitle}>
          <FileJson size={16} style={{ color: '#22d3ee' }} />
          Profile Format (miStudio Compatible)
        </div>
        <pre style={{ 
          background: 'rgba(30, 41, 59, 0.5)', 
          padding: '16px', 
          borderRadius: '8px', 
          fontSize: '12px',
          fontFamily: "'JetBrains Mono', monospace",
          color: '#94a3b8',
          overflow: 'auto',
          margin: 0
        }}>
{`{
  "name": "yelling-demo",
  "model": "google/gemma-2-2b",
  "sae": {
    "repo": "google/gemma-scope-2b-pt-res",
    "layer": 12
  },
  "features": [
    { "index": 1234, "strength": 5.0, "label": "Yelling/Capitalization" }
  ],
  "version": "1.0"
}`}
        </pre>
      </div>
    </div>
  );
};

const MonitorPage = ({ features }) => {
  const [monitoredFeatures, setMonitoredFeatures] = useState([1234, 892, 2341]);
  const [isLive, setIsLive] = useState(true);
  
  return (
    <div>
      <div style={styles.pageHeader}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={styles.pageTitle}>Feature Monitor</h1>
            <p style={styles.pageSubtitle}>Real-time observation of feature activations during inference</p>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              style={isLive ? { 
                ...styles.actionButton, 
                background: 'rgba(239, 68, 68, 0.1)', 
                borderColor: 'rgba(239, 68, 68, 0.3)', 
                color: '#f87171' 
              } : styles.buttonPrimary}
              onClick={() => setIsLive(!isLive)}
            >
              {isLive ? <><Square size={14} /> Pause</> : <><Play size={14} /> Resume</>}
            </button>
            <button style={styles.buttonSecondary}>
              <Settings size={14} /> Configure Features
            </button>
          </div>
        </div>
      </div>
      
      <div style={styles.monitorGrid}>
        <div style={styles.monitorCard}>
          <div style={styles.monitorValue}>247</div>
          <div style={styles.monitorLabel}>Requests/min</div>
        </div>
        <div style={styles.monitorCard}>
          <div style={styles.monitorValue}>142ms</div>
          <div style={styles.monitorLabel}>Avg Latency</div>
        </div>
        <div style={styles.monitorCard}>
          <div style={styles.monitorValue}>3</div>
          <div style={styles.monitorLabel}>Features Monitored</div>
        </div>
        <div style={styles.monitorCard}>
          <div style={{ ...styles.monitorValue, color: '#4ade80' }}>●</div>
          <div style={styles.monitorLabel}>{isLive ? 'Live' : 'Paused'}</div>
        </div>
      </div>
      
      <div style={styles.card}>
        <div style={styles.cardTitle}>
          <Activity size={16} style={{ color: '#22d3ee' }} />
          Monitored Features
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          {monitoredFeatures.map((featureIndex) => {
            const feature = features.find(f => f.index === featureIndex);
            if (!feature) return null;
            
            return (
              <div key={featureIndex} style={{ 
                background: 'rgba(30, 41, 59, 0.5)', 
                borderRadius: '10px', 
                padding: '20px',
                border: '1px solid rgba(71, 85, 105, 0.3)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <div>
                    <span style={{ ...styles.featureIndex, fontSize: '14px' }}>#{feature.index}</span>
                    <span style={{ ...styles.featureLabel, display: 'block', marginLeft: 0, marginTop: '4px' }}>{feature.label}</span>
                  </div>
                  <button style={styles.iconButton}>
                    <EyeOff size={14} />
                  </button>
                </div>
                <div style={{ 
                  fontSize: '36px', 
                  fontWeight: '700', 
                  fontFamily: "'JetBrains Mono', monospace",
                  color: '#f1f5f9',
                  marginBottom: '8px'
                }}>
                  {feature.activation.toFixed(3)}
                </div>
                <div style={styles.activationBar}>
                  <div style={{ 
                    ...styles.activationFill, 
                    width: `${feature.activation * 100}%`,
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  marginTop: '8px',
                  fontSize: '11px',
                  color: '#64748b'
                }}>
                  <span>Min: 0.12</span>
                  <span>Max: 0.95</span>
                  <span>Avg: 0.54</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      <div style={styles.card}>
        <div style={styles.cardTitle}>
          <Activity size={16} style={{ color: '#22d3ee' }} />
          Recent Activations Log
        </div>
        
        <table style={styles.table}>
          <thead style={styles.tableHeader}>
            <tr>
              <th style={styles.tableHeaderCell}>Timestamp</th>
              <th style={styles.tableHeaderCell}>Request ID</th>
              {monitoredFeatures.map(idx => (
                <th key={idx} style={styles.tableHeaderCell}>#{idx}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {mockMonitorData.map((row, i) => (
              <tr key={i}>
                <td style={{ ...styles.tableCell, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px' }}>{row.time}</td>
                <td style={{ ...styles.tableCell, fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', color: '#64748b' }}>req_{1000 + i}</td>
                {monitoredFeatures.map(idx => (
                  <td key={idx} style={{ ...styles.tableCell, fontFamily: "'JetBrains Mono', monospace" }}>
                    {(row.features[idx] || Math.random()).toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Main App
export default function MiLLMApp() {
  const [activeTab, setActiveTab] = useState('models');
  const [models, setModels] = useState(mockModels);
  const [saes, setSAEs] = useState(mockSAEs);
  const [profiles, setProfiles] = useState(mockProfiles);
  const [features] = useState(mockFeatures);
  
  const loadedModel = models.find(m => m.status === 'loaded');
  const attachedSAE = saes.find(s => s.status === 'attached');
  const activeProfile = profiles.find(p => p.active);
  
  const handleLoadModel = (modelName) => {
    setModels(models.map(m => ({
      ...m,
      status: m.name === modelName ? 'loaded' : (m.status === 'loaded' ? 'ready' : m.status)
    })));
  };
  
  const handleAttachSAE = (saeName) => {
    setSAEs(saes.map(s => ({
      ...s,
      status: s.name === saeName ? 'attached' : (s.status === 'attached' ? 'ready' : s.status)
    })));
  };
  
  const handleActivateProfile = (profileId) => {
    setProfiles(profiles.map(p => ({
      ...p,
      active: p.id === profileId
    })));
  };
  
  const tabs = [
    { id: 'models', label: 'Models' },
    { id: 'saes', label: 'SAEs' },
    { id: 'steering', label: 'Steering' },
    { id: 'profiles', label: 'Profiles' },
    { id: 'monitor', label: 'Monitor' },
  ];
  
  return (
    <div style={styles.app}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        input:focus, select:focus, button:focus { outline: none; }
        input:focus, select:focus { border-color: rgba(34, 211, 238, 0.5) !important; }
        
        button:hover { opacity: 0.9; }
        
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: #22d3ee;
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 2px 6px rgba(34, 211, 238, 0.4);
        }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(30, 41, 59, 0.3); border-radius: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(71, 85, 105, 0.5); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(71, 85, 105, 0.7); }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
      
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={styles.logo}>
              <div style={styles.logoIcon}>
                <Zap size={20} color="#fff" />
              </div>
              <div>
                <div style={styles.logoText}>miLLM</div>
                <div style={styles.logoSubtext}>Mechanistic Interpretability Server</div>
              </div>
            </div>
            
            <nav style={{ ...styles.nav, marginLeft: '48px' }}>
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  style={{
                    ...styles.navItem,
                    ...(activeTab === tab.id ? styles.navItemActive : {})
                  }}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
          
          <StatusBar 
            model={loadedModel?.name} 
            sae={attachedSAE?.name}
            profile={activeProfile?.name}
          />
        </div>
      </header>
      
      <main style={styles.main}>
        {activeTab === 'models' && (
          <ModelsPage models={models} onLoadModel={handleLoadModel} />
        )}
        {activeTab === 'saes' && (
          <SAEsPage saes={saes} models={models} onAttachSAE={handleAttachSAE} />
        )}
        {activeTab === 'steering' && (
          <SteeringPage saes={saes} features={features} />
        )}
        {activeTab === 'profiles' && (
          <ProfilesPage profiles={profiles} onActivateProfile={handleActivateProfile} />
        )}
        {activeTab === 'monitor' && (
          <MonitorPage features={features} />
        )}
      </main>
    </div>
  );
}
