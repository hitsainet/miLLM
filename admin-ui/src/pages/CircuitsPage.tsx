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
    acknowledgeUnvalidated: boolean;
  } | null>(null);
  // F19 R2-02: remember the ACKNOWLEDGEMENT too, not just which circuit.
  //
  // "Compose anyway" retried with `{circuitId, allowLayerOverlap: true}` and
  // dropped `acknowledgeUnvalidated`. So an operator who ticked the box on an
  // unvalidated circuit, then hit a contention refusal, was refused AGAIN with
  // UNVALIDATED_CIRCUIT — and the hook's handler toasts "tick the
  // acknowledgement to steer with it anyway", which they already did. The
  // dialog has closed, the checkbox lives in CircuitCard, and there is no path
  // forward: the override was unreachable for exactly the circuits where
  // composition is riskiest.
  const [pending, setPending] = useState<{
    circuitId: string;
    acknowledgeUnvalidated: boolean;
  } | null>(null);

  // F19 R3-18: circuits already sacrificed to this activation attempt.
  //
  // "Deactivate incumbent" now retries, and the retry can be refused by a
  // DIFFERENT incumbent — so the dialog reopens, looking identical to the
  // first time. The loop is bounded only by the number of distinct incumbents,
  // and every pass destroys a live circuit. Without a record, the operator
  // cannot tell iteration three from iteration one, or notice they have now
  // stopped three circuits to start one.
  const [sacrificed, setSacrificed] = useState<string[]>([]);

  const handleContention = useCallback(
    (details: ContentionDetails, message: string) => {
      setContention((prev) => ({
        details,
        message,
        // R2-03: an empty circuitId here would POST to `/circuits//activate`.
        // Prefer the pending click, fall back to whatever the previous
        // refusal named, and never invent one.
        circuitId: pending?.circuitId ?? prev?.circuitId ?? '',
        acknowledgeUnvalidated:
          pending?.acknowledgeUnvalidated ?? prev?.acknowledgeUnvalidated ?? false,
      }));
    },
    [pending],
  );

  const {
    circuits,
    isLoading,
    importCircuit,
    isImporting,
    activateCircuitQuiet,
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
                // R2-14: which of THIS circuit's layers are shared. Derived
                // from the claims the strip already fetches, so the card and
                // the strip cannot disagree.
                composedLayers={claims
                  .filter((c) => c.circuit_id === circuit.id && c.composed)
                  .map((c) => c.layer)}
                isActivating={isActivating}
                isDeactivating={isDeactivating}
                onActivate={(ack) => {
                  // A contention refusal arrives asynchronously, so "Compose
                  // anyway" needs BOTH which circuit and whether the operator
                  // already acknowledged an unvalidated rung.
                  setPending({
                    circuitId: circuit.id,
                    acknowledgeUnvalidated: ack,
                  });
                  // R2-05: the NON-throwing surface. `activateCircuit` is
                  // `mutateAsync`, which rethrows after `onError` — so every
                  // handled refusal (contention, unvalidated) also produced an
                  // UNHANDLED PROMISE REJECTION. `void` suppresses the lint,
                  // not the rejection.
                  activateCircuitQuiet({
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
          previouslyDeactivated={sacrificed}
          onCancel={() => {
            setContention(null);
            setSacrificed([]);
            // R2-03: clear the pending click too. It was set on every Activate
            // and never reset, so a refusal arriving after an unrelated later
            // click could compose the WRONG circuit.
            setPending(null);
          }}
          onDeactivateIncumbent={
            contention.details.incumbent?.id
              ? () => {
                  // F19 R2-15: deactivate AND retry. This closed the dialog
                  // and stopped, so the operator was left with nothing
                  // steering and had to find and re-click Activate — the
                  // two-step remedy the dialog offers had no completion step,
                  // and the reason they opened it was to get this circuit
                  // serving.
                  const incumbentId = contention.details.incumbent.id as string;
                  const label = contention.details.incumbent.name ?? incumbentId;
                  const { circuitId, acknowledgeUnvalidated } = contention;
                  setContention(null);
                  setPending(null);
                  setSacrificed((prev) =>
                    prev.includes(label) ? prev : [...prev, label],
                  );
                  void deactivateCircuit(incumbentId)
                    .then(() => {
                      activateCircuitQuiet({
                        circuitId,
                        acknowledgeUnvalidated,
                      });
                    })
                    .catch(() => {
                      // The deactivate mutation already toasts its failure.
                      // Do NOT retry: the layer is still held, and a second
                      // refusal here would read as the remedy having done
                      // nothing rather than as the deactivation having failed.
                    });
                }
              : undefined
          }
          onComposeAnyway={
            contention.details.overridable
              ? () => {
                  const { circuitId, acknowledgeUnvalidated } = contention;
                  setContention(null);
                  setPending(null);
                  setSacrificed([]);
                  activateCircuitQuiet({
                    circuitId,
                    // R2-02: carry the acknowledgement through, or the retry
                    // is refused for a reason the operator already resolved.
                    acknowledgeUnvalidated,
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
