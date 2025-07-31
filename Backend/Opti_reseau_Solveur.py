import pandas as pd
import geopandas as gpd
import numpy as np
import pulp
from shapely.geometry import Point
from geopy.distance import geodesic
from pulp import HiGHS
from sklearn.preprocessing import MinMaxScaler

def geodesic_distance(p1, p2):
    # p1 et p2 sont des tuples (lon, lat)
    return geodesic((p1[1], p1[0]), (p2[1], p2[0])).km

def convert_to_numeric_safe(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def filtrer_sites_proches(zones, sites, rayon_km=200):
    
    sites_filtres = set()

    for i in zones:
        for j in sites:
            d = geodesic(zones[i]['coord'], sites[j]['coord']).km
            if d <= rayon_km:
                sites_filtres.add(j)

    return {j: sites[j] for j in sites_filtres}

def Opti_reseau_Solveur (data,nuts_data, nb_entrepot, maille, Col_NUTS, ponderations = None) :
    
    # 1. Charger la base des codes postaux et des régions NUTS

    # nuts_data = pd.read_excel(r'C:\Users\AlbinGEFFRAY\OneDrive - Bartle Business Consulting\Documents\00.Datasupply\API\export_nuts.xlsx')
    
    # Charger le GeoJSON des régions NUTS -> pour l'affichage 
    geojson_data = gpd.read_file(r'data/NUTS_RG_10M_2024_4326.geojson')
    # Filtrer uniquement les NUTS de la maille souhaitée
    liste_maille = ["Pays", "NUTS_1", "NUTS_2", "NUTS_3", "IRIS"]
    rang = liste_maille.index(maille)
    geojson_data = geojson_data[geojson_data['LEVL_CODE'] == rang]

    
    # Jointure entre les commandes et les régions NUTS
    data['NUTS_3_2024'] = data[Col_NUTS].astype(str) 
    data = data.merge(nuts_data, on="NUTS_3_2024", how="left")


    # 2.b Gestion des pondérations 

    if ponderations is None:
        ponderations = []

    params = [param for param, _ in ponderations]        # extraire les noms de colonnes
    data   = convert_to_numeric_safe(data, params)       # conversion prudente en numérique
    data   = data.dropna(subset=params)                  # éliminer les NaN qui bloqueraient la somme
    agg_dict = {param: "sum" for param in params}        # sommation pour chaque col. pondérée  

    # Latitude / longitude : on en garde une seule valeur par maille
    agg_dict[maille + "_latitude"]  = "first"
    agg_dict[maille + "_longitude"] = "first"               
    
    data = (
        data
        .groupby(maille + "_2024", as_index=False)
        .agg(agg_dict)
    )
    print(ponderations)
    
    if ponderations:
        params = [param for param, _ in ponderations if param in data.columns]
        scaler = MinMaxScaler()
        data[params] = scaler.fit_transform(data[params])
        print(data.head())

    # --- 3. Construction des zones (demandes) ---
    zones = {}
    for idx, row in data.iterrows():
        try:
            lat = row[maille + "_latitude"]
            lon = row[maille + "_longitude"]
            if not np.isnan(lat) and not np.isnan(lon):
                if ponderations :
                    demande = sum(poids/100 * row[param] for param, poids in ponderations)
                else :
                    demande = 1
                zones[idx] = {
                    'coord': (lon, lat),  # attention : (lon, lat)
                    'demande': demande                 }
        except KeyError:
            continue

    # --- 4. Construction des sites candidats à l’implantation (dans nuts_data) ---
    sites = {}

    # Filtrer les lignes valides
    nuts_valides = nuts_data.dropna(subset=[maille + "_latitude", maille + "_longitude"])

    # Prendre la première ligne par groupe (valeurs identiques dans chaque groupe)
    grouped = nuts_valides.groupby(maille + "_2024").first().reset_index()

    # Construire le dictionnaire des sites
    for _, row in grouped.iterrows():
        code_site = row[maille + "_2024"]
        lat = row[maille + "_latitude"]
        lon = row[maille + "_longitude"]
        sites[code_site] = {'coord': (lon, lat)}
    
    # Avant : sites = tous les NUTS_3
    print(f"Nombre total de sites initiaux : {len(sites)}")

    # Filtrer les sites à moins de 300 km d'au moins une zone
    sites = filtrer_sites_proches(zones, sites, rayon_km=200)

    print(f"Nombre de sites filtrés : {len(sites)}")


    # --- 5. Calcul des distances  ---
    distances = {}
    for i in zones:
        for j in sites:
            distances[(i, j)] = geodesic_distance(zones[i]['coord'], sites[j]['coord'])


     # --- 6. Définition du problème d’optimisation ---
    prob = pulp.LpProblem("k_median_problem", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", ((i, j) for i in zones for j in sites), cat='Binary')
    y = pulp.LpVariable.dicts("y", (j for j in sites), cat='Binary')

    # Fonction objectif
    prob += pulp.lpSum(zones[i]['demande'] * distances[(i, j)] * x[(i, j)] for i in zones for j in sites)

    # Contraintes
    prob += pulp.lpSum(y[j] for j in sites) == nb_entrepot
    for i in zones:
        prob += pulp.lpSum(x[(i, j)] for j in sites) == 1 # Chaque zone est affceté à un seul entrepôt
        for j in sites:
            prob += x[(i, j)] <= y[j] # Une zone ne peut être affectée qu’à un entrepôt ouvert, pas possible d'affecter à la zone si pas d'entrepôt (y=0)

    
     # --- 7. Résolution ---
    prob.solve(HiGHS())
    print(f"Statut de la résolution : {pulp.LpStatus[prob.status]}")

    # --- 8. Résultats ---
    entrepots_ouverts = [j for j in sites if pulp.value(y[j]) and pulp.value(y[j]) > 0.5]
       
    # Récupération du code NUTS réel associé à chaque zone (index → code)
    zone_ids = data[maille + "_2024"].tolist()

    # Construction du dictionnaire de correspondance : code zone → entrepôt (code NUTS)
    affectations_nuts = {}
    for i in zones:
        min_dist = float('inf')
        closest_site = None
        for j in entrepots_ouverts:
            d = geodesic_distance(zones[i]['coord'], sites[j]['coord'])
            if d < min_dist:
                min_dist = d
                closest_site = j
        affectations_nuts[zone_ids[i]] = closest_site
    



    # Associe chaque zone de demande à son entrepôt (NUTS_2024 de l'entrepôt ouvert)
    data["Cluster"] = data[maille + "_2024"].map(affectations_nuts)

    # Merge avec les données géographiques pour récupérer la géométrie
    data = data.merge(geojson_data, left_on=maille + "_2024", right_on="NUTS_ID", how="left")
    colonnes_finales = [maille+"_2024", "Cluster"] + params + [maille + "_latitude", maille + "_longitude", "geometry"]


    # Création du DataFrame des entrepôts ouverts
    df_entrepots = pd.DataFrame({
        maille + "_2024": entrepots_ouverts,
        "x": [sites[j]["coord"][0] for j in entrepots_ouverts],
        "y": [sites[j]["coord"][1] for j in entrepots_ouverts],
        "Cluster": entrepots_ouverts  # Ici, le cluster = identifiant NUTS
    })
    print(df_entrepots)
    print(data[colonnes_finales])

    return df_entrepots, data[colonnes_finales]

