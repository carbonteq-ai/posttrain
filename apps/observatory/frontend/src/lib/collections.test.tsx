import { describe, expect, it } from 'vitest';

import { collectionAxis } from './collections';

describe('collectionAxis', () => {
  it('keeps steps as collections when every update has its own collection', () => {
    const axis = collectionAxis([]);
    expect(axis.remapped).toBe(false);
    expect(axis.position(7, 'update')).toBe(7);
    expect(axis.position(7, 'collection')).toBe(7);
    expect(axis.label(7)).toBe('Step 7');
  });

  it('numbers collections and places their updates leading up to them', () => {
    // Four updates per collection; the third collection has started (updates 9 and 10 recorded).
    const axis = collectionAxis([1, 5, 9]);
    expect(axis.remapped).toBe(true);
    expect([1, 5, 9].map((step) => axis.position(step, 'collection'))).toEqual([1, 2, 3]);
    expect([1, 2, 3, 4, 5, 8, 9, 10].map((step) => axis.position(step, 'update'))).toEqual([0.25, 0.5, 0.75, 1, 1.25, 2, 2.25, 2.5]);
    expect(axis.collection(6)).toBe(2);
    expect(axis.collection(10)).toBe(3);
    expect(axis.label(2)).toBe('Step 2');
    expect(axis.label(2.25)).toBe('Step 3 · update 1 of 4');
    expect(axis.label(1.75)).toBe('Step 2 · update 3 of 4');
  });
});
