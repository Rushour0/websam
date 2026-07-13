/**
 * Model-tier dropdown for the Toolbar's model-status cluster. No props —
 * reads/writes the store directly (matches Toolbar.tsx's own convention).
 * Lists every `listModels()` registry entry; selecting a tier that requires
 * license acceptance is gated by `LicenseConsentDialog` via the store's
 * `selectModel` action.
 */
import { listModels } from '@websam3/core';

import { useStudioStore } from '../store/studio-store.js';

export function ModelPicker(): React.JSX.Element {
  const selectedModelId = useStudioStore((s) => s.selectedModelId);
  const modelStatus = useStudioStore((s) => s.modelStatus);
  const selectModel = useStudioStore((s) => s.selectModel);
  const models = listModels();

  return (
    <select
      aria-label="Model tier"
      className="h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground disabled:opacity-50"
      value={selectedModelId}
      disabled={modelStatus.phase === 'loading'}
      onChange={(e) => selectModel(e.target.value)}
    >
      {models.map((spec) => (
        <option key={spec.id} value={spec.id}>
          {spec.displayName}
          {spec.license === 'sam-license' ? ' (SAM License)' : ''}
        </option>
      ))}
    </select>
  );
}
