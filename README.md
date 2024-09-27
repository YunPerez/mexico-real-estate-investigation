# mds-research-stay
Master in DS research stay in Mexico Housing Prices


# Setup :hammer:

## Python environment
To create the python environment, you can run the following command:

```bash
# python 3.10 and install the requirements
conda create -n mds-research-stay python=3.10
conda activate mds-research-stay
pip install -r requirements.txt
```

## INEGI API
For running the `InegiPY` package, you need to have two tokens from the INEGI API. You can get them by following these steps:

1. Go to the [INEGI API page](https://www.inegi.org.mx/servicios/api_indicadores.html)
    a. Get the **DENUE** token
    b. Get the **Ruteo** token
2. You must save them into a `.env` file in the root of the notebook project.

