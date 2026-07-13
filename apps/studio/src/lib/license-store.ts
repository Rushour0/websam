/**
 * Tiny localStorage-backed persistence for accepted license ids.
 *
 * No dependencies beyond the DOM. Used by the license-acceptance gate
 * (studio-store.ts's `selectModel`/`confirmPendingLicense`) so a previously
 * accepted license tier isn't re-prompted on subsequent visits to this origin.
 */

const STORAGE_PREFIX = 'websam:license-accepted:';

/** Whether `license` (e.g. 'sam') was previously accepted on this origin. Best-effort: false on any storage error (private browsing, quota, SSR). */
export function isLicenseAccepted(license: string): boolean {
  try {
    return window.localStorage.getItem(STORAGE_PREFIX + license) === '1';
  } catch {
    return false;
  }
}

/** Persist that `license` was accepted. Best-effort; swallows storage errors. */
export function acceptLicense(license: string): void {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + license, '1');
  } catch {
    // ignore — private browsing / quota exceeded; the in-session store state still reflects acceptance
  }
}
