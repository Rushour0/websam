import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge conditional class names (clsx) then dedupe conflicting Tailwind
 * utility classes (tailwind-merge). Standard shadcn/ui `cn()` helper.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
