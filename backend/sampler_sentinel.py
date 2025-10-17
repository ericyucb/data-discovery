import streamlit as st
import ee, json, os, random, time, requests
import pandas as pd
import folium
from io import BytesIO
from PIL import Image as PILImage
from streamlit_folium import st_folium
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------
# Earth Engine init
# -------------------------------
try:
    ee.Initialize(project="original-dahlia-471603-f2")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="original-dahlia-471603-f2")

# -------------------------------
# UI setup
# -------------------------------
st.set_page_config(page_title="AlphaEarth Sampler", layout="wide")
st.title("🛰️ AlphaEarth Caption Sampler (Sentinel-2 + Spinner)")
st.caption("Random Sentinel-2 thumbnails with cloud masking and AlphaEarth embeddings. Works across the contiguous U.S.")

# -------------------------------
# Sidebar parameters
# -------------------------------
st.sidebar.header("Parameters")
STATE = st.sidebar.selectbox(
    "State",
    ["ALL (US-wide)", "CA", "TX", "WA", "CO", "AZ", "NV", "OR", "UT", "NM", "WY", "MT", "ID"]
)
THUMB_METERS = st.sidebar.number_input("Half-size of thumbnail (m)", 200, 3000, 1000, step=100)
SAMPLE_SCALE = st.sidebar.number_input("Sample scale (m)", 20, 1000, 100)
THUMB_PIXELS = st.sidebar.slider("Thumbnail resolution (px)", 128, 1024, 512, step=64)
YEAR = st.sidebar.slider("Year", 2018, 2023, 2021)

square_area = 4 * (THUMB_METERS ** 2)
est_embeddings = int(square_area / (SAMPLE_SCALE ** 2))
st.sidebar.markdown(f"**Est. AlphaEarth vectors per caption:** {est_embeddings:,}")

# -------------------------------
# Data paths
# -------------------------------
os.makedirs("labels", exist_ok=True)
JSON_PATH = "labels/alphaearth_captions.json"
PARQUET_PATH = "labels/alphaearth_dataset.parquet"

records = []
if os.path.exists(JSON_PATH):
    try:
        records = json.load(open(JSON_PATH))
    except Exception:
        pass

if os.path.exists(PARQUET_PATH) and os.path.getsize(PARQUET_PATH) > 0:
    df = pq.read_table(PARQUET_PATH).to_pandas()
else:
    df = pd.DataFrame(columns=["lon", "lat", "caption", "thumb_m", "sample_scale"])

# -------------------------------
# Imagery setup (Sentinel-2 SR)
# -------------------------------
us_states = ee.FeatureCollection("TIGER/2018/States")
if STATE == "ALL (US-wide)":
    roi = us_states.filter(ee.Filter.notEquals("STUSPS","AK")) \
                   .filter(ee.Filter.notEquals("STUSPS","HI")).geometry()
else:
    roi = us_states.filter(ee.Filter.eq("STUSPS", STATE)).geometry()

alpha_img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
             .filterDate(f"{YEAR}-01-01", f"{YEAR+1}-01-01")
             .mosaic().select(["A.*"]))

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate(f"{YEAR}-03-01", f"{YEAR+1}-01-01")
      .filterBounds(roi)
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)))

def mask_clouds(img):
    scl = img.select("SCL")
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask)

s2_rgb = s2.map(mask_clouds).median().select(["B4","B3","B2"])
band_names = [f"A{i:02d}" for i in range(64)]

# -------------------------------
# Helpers
# -------------------------------
def save_parquet(records, path):
    df_new = pd.DataFrame([
        {"lon":r["lon"], "lat":r["lat"], "caption":r["caption"],
         "thumb_m":r["thumb_m"], "sample_scale":r["sample_scale"],
         "alphaearth":r["alphaearth"]}
        for r in records
    ])
    pq.write_table(pa.Table.from_pandas(df_new, preserve_index=False), path)

def get_random_valid_point(max_tries=6):
    """Find random point with valid Sentinel coverage."""
    for _ in range(max_tries):
        seed = random.randint(0, 999999)
        pt = ee.FeatureCollection.randomPoints(roi, 1, seed).first().geometry()
        lon, lat = pt.coordinates().getInfo()
        square = pt.buffer(THUMB_METERS).bounds()
        val = s2_rgb.reduceRegion(ee.Reducer.mean(), square, 120).get("B4").getInfo()
        if val is not None:
            return pt, lon, lat, square
    return None, None, None, None

def fetch_valid_thumbnail(square, tries=4, timeout=8):
    """Fetch thumbnail and ensure it actually loads."""
    for _ in range(tries):
        try:
            url = s2_rgb.getThumbURL({
                "region": square,
                "dimensions": THUMB_PIXELS,
                "format": "png",
                "min": 0, "max": 3000
            })
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 10000:
                img = PILImage.open(BytesIO(r.content))
                return url, img
        except Exception:
            time.sleep(0.5)
    return None, None

# -------------------------------
# Landing page
# -------------------------------
if "initialized" not in st.session_state:
    st.markdown("## 🌍 AlphaEarth Caption Sampler (Sentinel-2)")
    st.write("Click **Start Sampling** to fetch your first Sentinel-2 image.")
    if st.button("🚀 Start Sampling"):
        st.session_state.initialized = True
        st.session_state.current = get_random_valid_point()
        st.rerun()
    st.stop()

# -------------------------------
# Session sampling loop
# -------------------------------
if "current" not in st.session_state:
    st.session_state.current = get_random_valid_point()

pt, lon, lat, square = st.session_state.current
if pt is None:
    st.error("Could not find visible imagery; try smaller thumbnail or larger scale.")
    st.stop()

with st.spinner("⏳ Fetching Sentinel-2 thumbnail..."):
    url, img = fetch_valid_thumbnail(square)
    time.sleep(0.3)

if img is None:
    st.warning("⚠️ Could not load image (likely rate-limited). Retrying automatically...")
    st.session_state.current = get_random_valid_point()
    st.rerun()

st.success("✅ Image fetched successfully!")
st.image(img, caption=f"{STATE} — ({lat:.3f}, {lon:.3f})")

# -------------------------------
# Caption + buttons
# -------------------------------
caption = st.text_input("Caption for this region:")
col1, col2 = st.columns(2)
save_btn = col1.button("💾 Submit & Next")
skip_btn = col2.button("⏭️ Skip / Random New")

if skip_btn:
    st.session_state.current = get_random_valid_point()
    st.rerun()

if save_btn:
    if caption.strip():
        samples = alpha_img.sample(region=square, scale=SAMPLE_SCALE,
                                   numPixels=100, geometries=True)
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
        save_parquet(records, PARQUET_PATH)
        st.success(f"✅ Saved {len(feats)} AlphaEarth vectors for '{caption}'")
        st.session_state.current = get_random_valid_point()
        st.rerun()
    else:
        st.warning("Enter a caption before submitting.")

# -------------------------------
# Dataset summary
# -------------------------------
if os.path.exists(PARQUET_PATH) and os.path.getsize(PARQUET_PATH) > 0:
    df = pq.read_table(PARQUET_PATH).to_pandas()
    summary_df = df.drop(columns=["alphaearth"])
    st.markdown("### 📊 Current Dataset Summary")
    st.dataframe(summary_df.tail(15), use_container_width=True)
    st.write(f"Total labeled vectors: **{len(df)}**")

    m = folium.Map(location=[39, -98], zoom_start=4)
    for _, r in summary_df.tail(200).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=4, color="#2b8a3e",
            fill=True, fill_opacity=0.8, popup=r["caption"]).add_to(m)
    st_folium(m, width=700, height=500)
