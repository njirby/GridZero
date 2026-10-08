import { useEffect, useState } from "react";
import { useStore } from "../store";
import { loadModels, loadCurrentModel, controls, type ModelCard } from "../api";

const VARIANT_LABELS: Record<string, string> = {
  default: "default", low: "low", medium: "medium", high: "high",
  xhigh: "xhigh", off: "no thinking",
};

export function ModelSelector() {
  const [cards, setCards] = useState<ModelCard[]>([]);
  const [loaded, setLoaded] = useState(false);
  const model = useStore((s) => s.model);
  const variant = useStore((s) => s.variant);
  const setModelLocal = useStore((s) => s.setModel);

  useEffect(() => {
    let live = true;
    (async () => {
      const [mcards, cur] = await Promise.all([loadModels(), loadCurrentModel()]);
      if (!live) return;
      if (mcards.length) { setCards(mcards); }
      if (cur) { setModelLocal(cur.model, cur.variant); }
      setLoaded(true);
    })();
    return () => { live = false; };
  }, [setModelLocal]);

  const selected = cards.find((c) => c.id === model);
  const variants = selected?.variants?.length ? selected.variants : ["default"];

  const chooseModel = (id: string) => {
    const card = cards.find((c) => c.id === id);
    const v = "default";
    setModelLocal(id, v);
    controls.setModel(id, v);
  };
  const chooseVariant = (v: string) => {
    setModelLocal(model, v);
    controls.setModel(model, v);
  };

  return (
    <div className="modelsel" data-testid="model-selector">
      <span className="msel-label">MODEL</span>
      <select
        data-testid="model-select"
        value={model}
        disabled={!loaded || cards.length === 0}
        onChange={(e) => chooseModel(e.target.value)}
        title="Switch the model operating the grid (live)"
      >
        {(cards.length ? cards : [{ id: model, name: model, variants: ["default"] }]).map((c) => (
          <option key={c.id} value={c.id}>{c.name}</option>
        ))}
      </select>
      {variants.length > 1 && (
        <>
          <span className="msel-label">EFFORT</span>
          <select
            data-testid="variant-select"
            value={variant}
            disabled={!loaded}
            onChange={(e) => chooseVariant(e.target.value)}
            title="Reasoning effort (thinking depth / speed)"
          >
            {variants.map((v) => (
              <option key={v} value={v}>{VARIANT_LABELS[v] || v}</option>
            ))}
          </select>
        </>
      )}
    </div>
  );
}
