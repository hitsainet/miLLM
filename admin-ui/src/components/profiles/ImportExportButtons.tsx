import { useRef, useState } from 'react';
import { Download, Upload, FileJson } from 'lucide-react';
import { Button, Modal } from '@components/common';
import type { ProfileExport } from '@/types';

interface ImportExportButtonsProps {
  onExport: (profileId: number) => Promise<ProfileExport | null>;
  onImport: (data: ProfileExport) => Promise<void>;
  profiles: Array<{ id: number; name: string }>;
  isExporting?: boolean;
  isImporting?: boolean;
}

export function ImportExportButtons({
  onExport,
  onImport,
  profiles,
  isExporting,
  isImporting,
}: ImportExportButtonsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [selectedProfileId, setSelectedProfileId] = useState<number | null>(null);
  const [importError, setImportError] = useState<string | null>(null);

  const handleExportClick = () => {
    if (profiles.length === 1) {
      handleExport(profiles[0].id);
    } else {
      setShowExportModal(true);
    }
  };

  const handleExport = async (profileId: number) => {
    try {
      const data = await onExport(profileId);
      if (!data) {
        console.error('Export failed: no data returned');
        return;
      }
      const profile = profiles.find((p) => p.id === profileId);
      const filename = `${profile?.name || 'profile'}.json`.replace(/[^a-z0-9_-]/gi, '_');

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setShowExportModal(false);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  const handleImportClick = () => {
    setImportError(null);
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text) as ProfileExport;

      // Basic validation
      if (!data.name || !data.version) {
        throw new Error('Invalid profile format: missing required fields');
      }

      await onImport(data);
      setImportError(null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : 'Failed to import profile');
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>
      <div className="flex items-center gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={handleImportClick}
          loading={isImporting}
          leftIcon={<Upload className="w-4 h-4" />}
        >
          Import
        </Button>

        <Button
          variant="secondary"
          size="sm"
          onClick={handleExportClick}
          loading={isExporting}
          disabled={profiles.length === 0}
          leftIcon={<Download className="w-4 h-4" />}
        >
          Export
        </Button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {importError && (
        <p className="text-sm text-red-400 mt-2">{importError}</p>
      )}

      {/* Export Profile Selection Modal */}
      <Modal
        id="export-profile-select"
        title="Export Profile"
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        footer={
          <>
            <Button variant="secondary" onClick={() => setShowExportModal(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={() => selectedProfileId && handleExport(selectedProfileId)}
              disabled={!selectedProfileId}
              loading={isExporting}
              leftIcon={<Download className="w-4 h-4" />}
            >
              Export
            </Button>
          </>
        }
      >
        <div className="space-y-2">
          <p className="text-sm text-slate-400 mb-4">
            Select a profile to export:
          </p>
          {profiles.map((profile) => (
            <label
              key={profile.id}
              className={`
                flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors
                ${selectedProfileId === profile.id
                  ? 'bg-primary-500/10 border-primary-500/30'
                  : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
                }
              `}
            >
              <input
                type="radio"
                name="exportProfile"
                value={profile.id}
                checked={selectedProfileId === profile.id}
                onChange={() => setSelectedProfileId(profile.id)}
                className="w-4 h-4 text-primary-500 bg-slate-800 border-slate-600 focus:ring-primary-500"
              />
              <FileJson className="w-4 h-4 text-slate-400" />
              <span className="text-sm text-slate-200">{profile.name}</span>
            </label>
          ))}
        </div>
      </Modal>
    </>
  );
}
