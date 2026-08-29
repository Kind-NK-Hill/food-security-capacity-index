# Data sources and reuse terms

This repository distributes only compact processed inputs needed to reproduce
the reported analysis. Raw API responses and third-party workbooks are not
committed. The code license does not override any source-data terms.

## World Bank World Development Indicators

- Dataset: World Development Indicators (WDI)
- Uses: cereal production, population, electricity access, rural population,
  female agricultural employment, arable land, and fertilizer consumption
- Snapshot access dates: 28-29 August 2026
- API: <https://api.worldbank.org/v2/>
- Dataset terms: <https://www.worldbank.org/en/about/legal/permissions>
- Summary terms: <https://data.worldbank.org/summary-terms-of-use>
- Default license: CC BY 4.0 unless indicator metadata states otherwise

Attribution: World Bank, World Development Indicators; original data providers
are recorded per indicator in
`data/processed/explanatory/feature_dictionary.csv`. Values in this repository
were filtered to the declared countries/years, reshaped, and joined by ISO3 and
year. The World Bank does not endorse this project.

## FAOSTAT

- Databases: Food Balances (2010-) and Suite of Food Security Indicators
- Uses: cereal production/import/export, PoU, and FIES
- Snapshot access date: 28 August 2026
- Food Balances catalog:
  <https://data.fao.org/catalog/iso/2f264bb6-1238-459a-bf8b-0e2d0a16804a>
- Food-security indicators catalog:
  <https://data.fao.org/catalog/dataset/955d6564-40a9-48b4-b51b-f19d65bb3539>
- Statistical Database Terms of Use:
  <https://www.fao.org/contact-us/terms/db-terms-of-use/en>
- Default license: CC BY 4.0 unless dataset metadata states otherwise

Attribution: FAO. 2026. FAOSTAT: Food Balances and Suite of Food Security
Indicators. Accessed 28 August 2026. License: CC BY 4.0. The project derives
cereal SSR as `production / (production + imports - exports) * 100`, retains
PoU `<2.5` as an interval, and excludes suppressed FIES values. FAO does not
endorse this project.

## Global Hunger Index

The public repository does not redistribute the GHI workbook or row-level GHI
data. The report retains only a cited aggregate sensitivity result. See the
methodology and download terms at:

- <https://www.globalhungerindex.org/methodology.html>
- <https://www.globalhungerindex.org/download/>

## Processed snapshots

The committed CSV files are modified/derived datasets. Their provenance,
transformations, coverage, and missing-value semantics are documented here,
in the report, and in the feature dictionary. Users refreshing the APIs should
expect values to change when source organizations revise historical series.
