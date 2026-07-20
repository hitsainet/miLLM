/**
 * CircuitsPage (Feature 13) — imported mistudio.circuit-definition/v1
 * documents: list with evidence rung, import, activate behind the
 * unvalidated gate, and the slice-fallback disclosure.
 */

import { useState } from 'react';
import { Plus, Share2 } from 'lucide-react';

import { useCircuits } from '@hooks/useCircuits';
import { CircuitCard, CircuitImportDialog } from '@components/circuits';
import { AttachmentPanel } from '@components/circuits';
import { Button, Card, EmptyState, Spinner } from '@components/common';

export function CircuitsPage() {
  const {
    circuits,
    isLoading,
    importCircuit,
    isImporting,
    activateCircuit,
    isActivating,
    deactivateCircuit,
    isDeactivating,
    deleteCircuit,
    exportCircuit,
  } = useCircuits();

  const [showImport, setShowImport] = useState(false);

  const handleExport = async (circuitId: string, name: string) => {
    const doc = await exportCircuit(circuitId);
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name.replace(/[^\w.-]+/g, '_')}.circuit.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <Share2 className="w-5 h-5 text-cyan-400" />
            Circuits
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Cross-layer circuits discovered and validated in miStudio. Each carries an
            evidence rung — only a causally validated circuit activates without an
            explicit acknowledgement.
          </p>
        </div>
        <Button variant="primary" onClick={() => setShowImport(true)}>
          <Plus className="w-4 h-4 mr-1" />
          Import
        </Button>
      </div>

      {/* Multi-SAE attachment set (Feature 12) — the SAEs a circuit serves through. */}
      <AttachmentPanel />

      <Card>
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Spinner size="lg" />
          </div>
        ) : circuits.length === 0 ? (
          <EmptyState
            icon={<Share2 className="w-6 h-6" />}
            title="No circuits imported"
            description="Import a circuit definition exported from miStudio to steer a cross-layer circuit here."
          />
        ) : (
          <div className="space-y-3">
            {circuits.map((circuit) => (
              <CircuitCard
                key={circuit.id}
                circuit={circuit}
                isActivating={isActivating}
                isDeactivating={isDeactivating}
                onActivate={(ack) =>
                  void activateCircuit({
                    circuitId: circuit.id,
                    acknowledgeUnvalidated: ack,
                  })
                }
                onDeactivate={() => void deactivateCircuit(circuit.id)}
                onDelete={() => void deleteCircuit(circuit.id)}
                onExport={() => void handleExport(circuit.id, circuit.name)}
              />
            ))}
          </div>
        )}
      </Card>

      <CircuitImportDialog
        open={showImport}
        onClose={() => setShowImport(false)}
        onImport={importCircuit}
        isImporting={isImporting}
      />
    </div>
  );
}

export default CircuitsPage;
