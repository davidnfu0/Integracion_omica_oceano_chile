# Integración de datos “ómicos” del océano costero chileno con la ayuda de aprendizaje de máquinas.

Se puede acceder a la pagina del proyecto en: [https://davidnfu0.github.io/Integracion_omica_oceano_chile/](https://davidnfu0.github.io/Integracion_omica_oceano_chile/)

La disponibilidad de datos "ómicos" en el océano costero chileno es limitada. Por lo que, es crucial integrar y analizar los datos existentes para comprender mejor los ecosistemas marinos. El aprendizaje de máquinas es una herramienta que puede ser muy util para obtener información valiosa a partir de estos datos.

Los datos con los que se trabaja en este proyecto son:
1. **Matriz de genes metabólicos:** Abundancia de genes o funciones metabólicas específicas.
2. **Matriz de genes asociados a ciclos biogeoquímicos:** Abundancia de genes
o familias génicas involucradas en procesos del ciclo del carbono, nitrógeno,
azufre, etc.
3. **Matriz de abundancia taxonómica a nivel de orden:** Abundancia de órdenes
taxonómicos.
4. **Matriz de abundancia taxonómica a nivel de filo:** Abundancia de filos microbianos.

Además, se cuenta con metadatos asociados a las muestras, con los cuales se puede enriquecer el análisis. Los principales metadatos se dividen en dos grupos:
1. **Metadatos ambientales:** Incluyen variables como temperatura, salinidad,
nutrientes, oxígeno disuelto, entre otros.
2. **Metadatos geográficos:** Incluyen información sobre la ubicación de las
muestras, como latitud, longitud y profundidad.

## Flujo de trabajo para el proyecto

Este proyecto sigue un flujo de trabajo estructurado pensado para ser reproducible y eficiente. La idea es que cualquier persona pueda replicar los análisis e integrar nuevos datos en el futuro. La ejecución del análisis se realiza a través de informes en **Quarto**, más adelante se detalla cómo crear el entorno y ejecutar los informes. El flujo de trabajo consta de los siguientes pasos:

1. **Preprocesamiento de datos y metadatos:** Limpieza y transformación de los datos crudos para asegurar su calidad y consistencia. Este es el paso que más puede variar dependiendo de los datos disponibles, en caso de integrar nuevos datos, este paso puede requerir modificaciones importantes.
2. **Normalización de datos:** Aplicación de técnicas de normalización para ajustar las diferencias en las escalas de los datos.
3. **Metodos de reducción de dimensionalidad:** Uso de técnicas como PCA, MOFA o UMAP para reducir la dimensionalidad de los datos. Este paso también puede requerir ajustes menores dependiendo de los datos.
4. **Aplicación de algoritmos de aprendizaje de máquinas:** Implementación de algoritmos supervisados y no supervisados para identificar patrones y relaciones en los datos.
5. **Visualización y análisis de resultados:** Creación de gráficos y tablas para interpretar los resultados obtenidos.

## Creando el entorno y ejecutando los informes

Descargar e instalar [Anaconda](https://www.anaconda.com/products/distribution) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) si no se tiene instalado.

Descargar el repositorio:
```bash
git clone https://github.com/davidnfu0/Integracion_omica_oceano_chile.git
cd Integracion_omica_oceano_chile
```

Crear el entorno de conda e instalar las dependencias:
```bash
conda env create -f environment.yml
conda activate Omic_Integration
python -m ipykernel install --user --name Omic_Integration --display-name "Omic_Integration"
```

Ejecutar los informes de Quarto:
```bash
quarto render
```
