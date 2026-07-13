/**
 * License-acceptance gate for license-restricted model tiers (e.g. SAM). Renders nothing when no
 * consent is pending; reads/writes `pendingLicenseModelId` directly from the studio store, matching
 * this repo's no-props, store-driven component convention (see ModelPicker.tsx / Toolbar.tsx).
 *
 * IMPORTANT: this dialog is a placeholder consent gate — it does not bundle the actual SAM license
 * text (see docs/PROGRESS.md). Do not fabricate real legal text here.
 */
import * as Dialog from '@radix-ui/react-dialog';
import { getModel } from '@websam3/core';

import { useStudioStore } from '../store/studio-store.js';

import { Button } from './ui/button.js';

export function LicenseConsentDialog(): JSX.Element | null {
  const pendingLicenseModelId = useStudioStore((s) => s.pendingLicenseModelId);
  const confirmPendingLicense = useStudioStore((s) => s.confirmPendingLicense);
  const cancelPendingLicense = useStudioStore((s) => s.cancelPendingLicense);

  if (!pendingLicenseModelId) return null;
  const spec = getModel(pendingLicenseModelId);

  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open) cancelPendingLicense();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-background p-6 text-foreground shadow-lg">
          <Dialog.Title className="text-sm font-semibold">License required</Dialog.Title>
          <Dialog.Description className="mt-2 text-xs text-muted-foreground">
            {spec?.displayName ?? pendingLicenseModelId} ships under Meta&apos;s SAM license. This
            build does not bundle the full license text yet — accepting here is a placeholder
            consent gate pending the real license text and model hosting (see docs/PROGRESS.md).
            Review the actual SAM license terms from Meta before using this tier for anything beyond
            local testing.
          </Dialog.Description>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={cancelPendingLicense}>
              Cancel
            </Button>
            <Button size="sm" onClick={confirmPendingLicense}>
              I Accept
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
