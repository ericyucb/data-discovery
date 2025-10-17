import streamlit as st, ee, folium, json, os
from streamlit_folium import st_folium
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

# init EE
try:
    ee.Initialize(project="original-dahlia-471603-f2")
except:
    ee.Authenticate()
    ee.Initialize(project="original-dahlia-471603-f2")

st.set_page_config(layout="wide")
st.title("🌍 AlphaEarth Explorer (Esri Stallie)")
st.caption("Pan & zoom the live satellite map. Click to sample AlphaEarth embeddings.")

THUMB_METERS = st.sidebar.number_input("Half-size of patch (m)", 200, 2000, 1000)
SAMPLE_SCALE = st.sidebar.number_input("Sample scale (m)", 10, 500, 100)
YEAR = st.sidebar.slider("Year", 2018, 2023, 2021)

# files
os.makedirs("labels", exist_ok=True)
PARQUET_PATH = "labels/alphaearth_clicks.parquet"
JSON_PATH = "labels/alphaearth_clicks.json"

records = []
if os.path.exists(JSON_PATH):
    try: records = json.load(open(JSON_PATH))
    except: pass

alpha_img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
             .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
             .mosaic().select(["A.*"]))
band_names = [f"A{i:02d}" for i in range(64)]

# --- interactive Google map ---
m = folium.Map(
    location=[39, -98],
    zoom_start=4,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery"
)

m.add_child(folium.LatLngPopup())
st.markdown("### 🗺️ Zoom & click anywhere to sample")
output = st_folium(m, width=1000, height=600)

if output and output.get("last_clicked"):
    lon = output["last_clicked"]["lng"]
    lat = output["last_clicked"]["lat"]
    st.markdown(f"**Selected point:** {lat:.5f}, {lon:.5f}")

    caption = st.text_input("Caption for this patch:", "")
    if st.button("💾 Save Sample"):
        pt = ee.Geometry.Point([lon, lat])
        square = pt.buffer(THUMB_METERS).bounds()
        samples = alpha_img.sample(region=square, scale=SAMPLE_SCALE, geometries=True)
        feats = samples.getInfo()["features"]
        for f in feats:
            vals = [f["properties"].get(b) for b in band_names]
            records.append({
                "lon": f["geometry"]["coordinates"][0],
                "lat": f["geometry"]["coordinates"][1],
                "caption": caption.strip(),
                "alphaearth": vals,
                "thumb_m": THUMB_METERS,
                "sample_scale": SAMPLE_SCALE
            })
        json.dump(records, open(JSON_PATH, "w"), indent=2)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(records)), PARQUET_PATH)
        st.success(f"✅ Saved {len(feats)} AlphaEarth vectors for '{caption}'")

if os.path.exists(PARQUET_PATH):
    df = pq.read_table(PARQUET_PATH).to_pandas()
    st.markdown("### 📊 Dataset Summary")
    st.write(df.tail(10))
