# Authored report presentation

These files preserve the report's authored Data app components outside ignored runtime state. They are used in the existing Data app report, not as a separate application or reporting backend.

To restore them, copy the JSX and CSS files into `.posttrain/state/analysis/sampo-multirun/report-app/src/content/report/`, keeping the installed Data app runtime and its public APIs. Run `build_report.py`, then the installed Data app `build` and `export-offline` commands documented in the research workflow. Export to `report.html` only after validation.

The presentation keeps original narrative block IDs, source query scopes, native MathML, retained trajectory projections and editable report behavior. Details disclosures preserve the complete analytical text while reducing the main reading path. The report title is owned by `build_report.py`; a browser's saved local presentation can retain an earlier title until edited through the report UI.
