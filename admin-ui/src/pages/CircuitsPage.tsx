/**
 * CircuitsPage (Feature 13) — imported mistudio.circuit-definition/v1
 * documents: list with evidence rung, import, activate behind the
 * unvalidated gate, and the slice-fallback disclosure.
 */

import { useCallback, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Plus, Share2 } from 'lucide-react';

import { useCircuits } from '@hooks/useCircuits';
import { circuitApi } from '@/services/circuits';
import { useToast } from '@hooks/useToast';
import {
  CircuitCard,
  CircuitImportDialog,
  ClaimsStrip,
  ContentionDialog,
  type ContentionDetails,
} from '@components/circuits';
import { AttachmentPanel, EdgeSensingPanel } from '@components/circuits';
import { Button, Card, EmptyState, Spinner } from '@components/common';

export function CircuitsPage() {
  // F19: the contention refusal is a decision point, so it gets a dialog with
  // both resolutions rather than a toast that discards the incumbent, the
  // measurement and the override route.
  const [contention, setContention] = useState<{
    details: ContentionDetails;
    message: string;
    circuitId: string;
  } | null>(null);
  const [pendingCircuitId, setPendingCircuitId] = useState<string | null>(null);

  const handleContention = useCallback(
    (details: ContentionDetails, message: string) => {
      setContention((prev) => ({
        details,
        message,
        circuitId: pendingCircuitId ?? prev?.circuitId ?? '',
      }));
    },
    [pendingCircuitId],
  );

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
  } = useCircuits(handleContention);

  const toast = useToast();
  const [showImport, setShowImport] = useState(false);

  // F19: who holds which layer. The unit of contention is the LAYER, so the
  // circuit list alone cannot answer it — two circuits can both be active
  // while contending for nothing. This makes a refusal intelligible BEFORE it
  // happens.
  const { data: claims = [] } = useQuery({
    queryKey: ['circuits', 'claims'],
    queryFn: () => circuitApi.claims(),
    refetchInterval: 15_000,
  });


  const handleExport = async (circuitId: string, name: string) => {
    let url: string | null = null;
    try {
      const doc = await exportCircuit(circuitId);
      const blob = new Blob([JSON.stringify(doc, null, 2)], {
        type: 'application/json',
      });
      url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.replace(/[^\w.-]+/g, '_')}.circuit.json`;
      a.click();
    } catch (error) {
      // Without this the click silently did nothing (a `void`-discarded
      // rejection) — every other circuit action reports its failure.
      toast.error(
        `Export failed: ${error instanceof Error ? error.message : 'unknown error'}`,
      );
    } finally {
      // Revoke in `finally` so a throw between create and click cannot pin the
      // blob (exports are the largest payloads in this feature).
      if (url) URL.revokeObjectURL(url);
    }
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
            <ClaimsStrip claims={claims} />
            {circuits.map((circuit) => (
              <CircuitCard
                key={circuit.id}
                circuit={circuit}
                isActivating={isActivating}
                isDeactivating={isDeactivating}
                onActivate={(ack) => {
                  // Remember WHICH circuit was being activated: a contention
                  // refusal arrives asynchronously, and "Compose anyway" has to
                  // retry the same one.
                  setPendingCircuitId(circuit.id);
                  void activateCircuit({
                    circuitId: circuit.id,
                    acknowledgeUnvalidated: ack,
                  });
                }}
                onDeactivate={() => void deactivateCircuit(circuit.id)}
                onDelete={() => void deleteCircuit(circuit.id)}
                onExport={() => void handleExport(circuit.id, circuit.name)}
              />
            ))}
          </div>
        )}
      </Card>

      {/* F19: the contention refusal, with both resolutions. A same-key
          collision renders no compose action — the dialog decides that from
          `overridable`, so the UI cannot offer an override the server would
          refuse anyway. */}
      {contention && (
        <ContentionDialog
          details={contention.details}
          message={contention.message}
          isBusy={isActivating || isDeactivating}
          onCancel={() => setContention(null)}
          onDeactivateIncumbent={
            contention.details.incumbent?.id
              ? () => {
                  const incumbentId = contention.details.incumbent.id as string;
                  setContention(null);
                  void deactivateCircuit(incumbentId);
                }
              : undefined
          }
          onComposeAnyway={
            contention.details.overridable
              ? () => {
                  const { circuitId } = contention;
                  setContention(null);
                  void activateCircuit({
                    circuitId,
                    allowLayerOverlap: true,
                  });
                }
              : undefined
          }
        />
      )}

      {/* Observed edge firings (Feature 15) — what the armed circuit's edges
          actually did on live traffic. */}
      <Card>
        <EdgeSensingPanel />
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
