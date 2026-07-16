/**
 * ClustersPage (Feature 8) — imported mistudio.cluster-definition/v1
 * documents as one-click steering profiles: list, import (paste/file/HF),
 * activate with the hard compatibility gate, dial lambda, export.
 */

import { useState } from 'react';
import { Boxes, Plus } from 'lucide-react';
import { useClusters } from '@hooks/useClusters';
import { ClusterCard, ClusterImportDialog } from '@components/clusters';
import { SensingPanel } from '@components/clusters/sensing';
import { useSensingToggle } from '@hooks/useSensing';
import { Button, Card, EmptyState, Spinner } from '@components/common';

export function ClustersPage() {
  const {
    clusters,
    isLoading,
    importClusters,
    isImporting,
    hubImport,
    activateCluster,
    isActivating,
    deactivateCluster,
    isDeactivating,
    setIntensity,
    exportCluster,
  } = useClusters();
  const { setEnabled: setSensingEnabled } = useSensingToggle();

  const [showImport, setShowImport] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <Boxes className="w-5 h-5 text-cyan-400" />
            Clusters
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Imported cluster definitions — tuned in miStudio or shared on Hugging Face —
            steer all members together at their authored strengths.
          </p>
        </div>
        <Button variant="primary" onClick={() => setShowImport(true)}>
          <Plus className="w-4 h-4 mr-1" />
          Import
        </Button>
      </div>

      <Card>
        {isLoading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : clusters.length === 0 ? (
          <EmptyState
            icon={<Boxes className="w-8 h-8" />}
            title="No clusters imported"
            description="Import a .cluster.json exported from miStudio, or browse public cluster packs on Hugging Face."
            action={
              <Button variant="primary" onClick={() => setShowImport(true)}>
                Import a cluster
              </Button>
            }
          />
        ) : (
          <div className="space-y-3">
            {clusters.map((c) => (
              <ClusterCard
                key={c.id}
                cluster={c}
                onActivate={() => activateCluster(c.id).catch(() => {})}
                onDeactivate={() => deactivateCluster(c.id).catch(() => {})}
                onSetIntensity={(intensity) => setIntensity({ id: c.id, intensity }).catch(() => {})}
                onExport={() => void exportCluster(c.id, c.name)}
                onToggleSensing={(enabled) => setSensingEnabled(c.id, enabled)}
                isActivating={isActivating}
                isDeactivating={isDeactivating}
              />
            ))}
          </div>
        )}
      </Card>

      <Card>
        <SensingPanel />
      </Card>

      <ClusterImportDialog
        open={showImport}
        onClose={() => setShowImport(false)}
        onImport={importClusters}
        onHubImport={hubImport}
        isImporting={isImporting}
      />
    </div>
  );
}
