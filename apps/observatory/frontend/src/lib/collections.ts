/**
 * Steps are collections. A collection is one sampled population of rollouts; a run may train on it with
 * one optimizer update (step = update = collection) or with several. When it uses several, the trainer
 * records population metrics once per collection at its first update's step and optimizer metrics once
 * per update, so raw steps count updates and runs with different update schedules stop lining up.
 *
 * This axis numbers collections 1..n. A collection's own metrics sit at its number; its updates sit at
 * fractions leading up to it (four updates of collection 3 at 2.25, 2.5, 2.75 and 3), so the last update
 * shares the collection's position exactly as it does in a one-update-per-collection run.
 */
export type CollectionAxis = {
  /** True when collections span several updates and raw steps are therefore remapped. */
  remapped: boolean;
  /** Axis position of a raw recorded step for a collection-grain or update-grain series. */
  position: (step: number, grain: 'collection' | 'update') => number;
  /** Collection number of a raw step (a rollout's step is its collection's first update). */
  collection: (step: number) => number;
  /** Human label for an axis position: "Step 3" or "Step 3 · update 2 of 4". */
  label: (position: number) => string;
};

const identity: CollectionAxis = {
  remapped: false,
  position: (step) => step,
  collection: (step) => step,
  label: (position) => `Step ${Number.isInteger(position) ? position : position.toFixed(2)}`,
};

export function collectionAxis(collectionSteps: readonly number[] | null | undefined): CollectionAxis {
  const starts = [...new Set(collectionSteps ?? [])].sort((left, right) => left - right);
  if (starts.length === 0) return identity;
  const spans = starts.slice(1).map((start, index) => start - starts[index]);
  // The newest collection has no successor yet; it uses the run's usual update count.
  const usualSpan = spans.length ? mode(spans) : 1;
  const spanOf = (index: number) => spans[index] ?? usualSpan;
  const indexOf = (step: number) => {
    let low = 0;
    let high = starts.length - 1;
    if (step < starts[0]) return 0;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (starts[middle] <= step) low = middle;
      else high = middle - 1;
    }
    return low;
  };
  return {
    remapped: true,
    position(step, grain) {
      const index = indexOf(step);
      if (grain === 'collection') return index + 1;
      const span = Math.max(spanOf(index), step - starts[index] + 1);
      return index + (step - starts[index] + 1) / span;
    },
    collection: (step) => indexOf(step) + 1,
    label(position) {
      const collection = Math.max(1, Math.ceil(position - 1e-9));
      const span = spanOf(collection - 1);
      const update = Math.round((position - (collection - 1)) * span);
      return update >= span || update <= 0
        ? `Step ${collection}`
        : `Step ${collection} · update ${update} of ${span}`;
    },
  };
}

function mode(values: readonly number[]): number {
  const counts = new Map<number, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort((left, right) => right[1] - left[1] || left[0] - right[0])[0][0];
}
