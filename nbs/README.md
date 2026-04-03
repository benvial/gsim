# Building Docs from Notebooks

Notebooks are committed **with outputs**. CI only converts and deploys — it does not re-execute them.

## Run a notebook for docs

```bash
just nbrun-docs nbs/my_notebook.ipynb
```

This executes the notebook with `PLOTLY_RENDERER=notebook_connected` (so Plotly produces self-contained HTML) and
`PYVISTA_OFF_SCREEN=true` (so PyVista renders static images instead of interactive widgets).

You can run multiple at once:

```bash
just nbrun-docs nbs/meep_ybranch.ipynb nbs/palace_demo_cpw.ipynb
```

## Build and preview docs locally

```bash
just nbdocs   # convert notebooks to markdown
just docs     # build mkdocs site
just serve    # serve at http://localhost:8080/gsim/
```

## Add a new notebook

1. Create `nbs/my_notebook.ipynb` and develop it normally
1. Run it for docs: `just nbrun-docs nbs/my_notebook.ipynb`
1. Add an entry in `mkdocs.yml` under `nav` pointing to `nbs/my_notebook.md`
1. Commit the executed notebook

## Update an existing notebook

1. Edit and run the notebook normally in Jupyter
1. Re-run for docs: `just nbrun-docs nbs/my_notebook.ipynb`
1. Commit
