import { Check, CaretDown } from '@phosphor-icons/react';
import * as Popover from '@radix-ui/react-popover';
import { useState } from 'react';

type FilterOption = {
  label: string;
  value: string;
};

type FilterPopoverProps = {
  label: string;
  value: string;
  options: FilterOption[];
  onChange: (value: string) => void;
};

export function FilterPopover({ label, value, options, onChange }: FilterPopoverProps) {
  const [open, setOpen] = useState(false);
  const selected = options.find((option) => option.value === value) ?? options[0];
  const active = value !== 'all';

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          data-testid={`filter-${label.toLowerCase().replaceAll(' ', '-')}`}
          className={`obs-control inline-flex h-8 items-center gap-2 px-2.5 transition-colors hover:border-[#cfc8d7] hover:text-ink ${
            active ? 'border-violet-300 bg-violet-50 text-violet-800' : ''
          }`}
          aria-label={`${label}: ${selected.label}`}
        >
          <span className="text-[10px] font-medium uppercase tracking-[.08em] text-muted">{label}</span>
          <span>{selected.label}</span>
          <CaretDown size={11} aria-hidden="true" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          sideOffset={6}
          align="start"
          className="z-50 min-w-[190px] rounded-md border border-divider bg-surface p-1.5 shadow-[0_12px_32px_rgba(40,35,44,.12)] outline-none"
        >
          <p className="px-2 pb-1.5 pt-1 text-[10px] font-medium uppercase tracking-[.1em] text-muted">
            Filter by {label.toLowerCase()}
          </p>
          <div role="listbox" aria-label={label} className="space-y-0.5">
            {options.map((option) => {
              const isSelected = option.value === value;
              return (
                <button
                  key={option.value}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => {
                    onChange(option.value);
                    setOpen(false);
                  }}
                  className={`flex w-full items-center justify-between rounded px-2 py-2 text-left text-xs ${
                    isSelected ? 'bg-violet-50 text-violet-800' : 'text-secondary hover:bg-subtle hover:text-ink'
                  }`}
                >
                  {option.label}
                  {isSelected && <Check size={13} weight="bold" aria-hidden="true" />}
                </button>
              );
            })}
          </div>
          <Popover.Arrow className="fill-surface" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
