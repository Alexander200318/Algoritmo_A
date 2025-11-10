import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import heapq

# Nodos de la ciudad de Cuenca
CUENCA_NODES = {
    "Catedral Nueva": {"lat": -2.8975, "lon": -79.005},
    "Parque Calderón": {"lat": -2.89741, "lon": -79.00438},
    "Puente Roto": {"lat": -2.90423, "lon": -79.00142},
    "Museo Pumapungo": {"lat": -2.90607, "lon": -78.99681},
    "Terminal Terrestre": {"lat": -2.89222, "lon": -78.99277},
    "Mirador de Turi": {"lat": -2.92583, "lon": -79.0040},
    "Universidad de Cuenca": {"lat": -2.90311, "lon": -79.00594},
    "Estadio Alejandro Serrano Aguilar": {"lat": -2.91167, "lon": -79.00833},
    "Barranco del Río Tomebamba": {"lat": -2.90194, "lon": -79.00083},
    "Mercado 10 de Agosto": {"lat": -2.89583, "lon": -79.00278},
    "Hospital Vicente Corral Moscoso": {"lat": -2.88944, "lon": -79.00472},
    "Aeropuerto Mariscal Lamar": {"lat": -2.88972, "lon": -78.98444},
    "Parque El Paraíso": {"lat": -2.88389, "lon": -78.99889},
    "Centro Comercial Mall del Río": {"lat": -2.91056, "lon": -78.99722},
    "Iglesia de San Sebastián": {"lat": -2.90139, "lon": -79.01056},
    "Parque de la Madre": {"lat": -2.88694, "lon": -79.00139},
}

# Conexiones del grafo
GRAPH_EDGES = {
    "Catedral Nueva": ["Parque Calderón", "Puente Roto", "Museo Pumapungo", "Universidad de Cuenca"],
    "Parque Calderón": ["Catedral Nueva", "Terminal Terrestre", "Puente Roto", "Mercado 10 de Agosto"],
    "Puente Roto": ["Catedral Nueva", "Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Barranco del Río Tomebamba"],
    "Museo Pumapungo": ["Catedral Nueva", "Puente Roto", "Terminal Terrestre", "Centro Comercial Mall del Río"],
    "Terminal Terrestre": ["Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Aeropuerto Mariscal Lamar"],
    "Mirador de Turi": ["Puente Roto", "Terminal Terrestre", "Estadio Alejandro Serrano Aguilar"],
    "Universidad de Cuenca": ["Catedral Nueva", "Barranco del Río Tomebamba", "Iglesia de San Sebastián"],
    "Estadio Alejandro Serrano Aguilar": ["Mirador de Turi", "Centro Comercial Mall del Río", "Iglesia de San Sebastián"],
    "Barranco del Río Tomebamba": ["Puente Roto", "Universidad de Cuenca", "Mercado 10 de Agosto"],
    "Mercado 10 de Agosto": ["Parque Calderón", "Barranco del Río Tomebamba", "Hospital Vicente Corral Moscoso"],
    "Hospital Vicente Corral Moscoso": ["Mercado 10 de Agosto", "Parque de la Madre", "Parque El Paraíso"],
    "Aeropuerto Mariscal Lamar": ["Terminal Terrestre", "Parque El Paraíso"],
    "Parque El Paraíso": ["Hospital Vicente Corral Moscoso", "Aeropuerto Mariscal Lamar", "Parque de la Madre"],
    "Centro Comercial Mall del Río": ["Museo Pumapungo", "Estadio Alejandro Serrano Aguilar"],
    "Iglesia de San Sebastián": ["Universidad de Cuenca", "Estadio Alejandro Serrano Aguilar"],
    "Parque de la Madre": ["Hospital Vicente Corral Moscoso", "Parque El Paraíso"],
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia haversine entre dos puntos en km"""
    R = 6371  # Radio de la Tierra en kilómetros
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def euclidean_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

class AStarPathFinder:
    def __init__(self, nodes, edges, distance_func=haversine_distance):
        self.nodes = nodes
        self.edges = edges
        self.distance_func = distance_func
    
    def heuristic(self, node1, node2):
        """Calcula la heurística (distancia estimada) entre dos nodos"""
        coords1 = self.nodes[node1]
        coords2 = self.nodes[node2]
        return self.distance_func(coords1["lat"], coords1["lon"], coords2["lat"], coords2["lon"])
    
    def get_distance(self, node1, node2):
        """Calcula la distancia real entre dos nodos conectados"""
        coords1 = self.nodes[node1]
        coords2 = self.nodes[node2]
        return self.distance_func(coords1["lat"], coords1["lon"], coords2["lat"], coords2["lon"])
    
    def find_path(self, start, goal):
        """Encuentra el camino más corto usando A*"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in self.nodes}
        f_score[start] = self.heuristic(start, goal)
        
        explored_nodes = []
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            explored_nodes.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, g_score[goal], explored_nodes
            
            for neighbor in self.edges.get(current, []):
                tentative_g_score = g_score[current] + self.get_distance(current, neighbor)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, None, explored_nodes

# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Rutas en Cuenca con A*", layout="wide")

st.title("🗺️ Planificador de Rutas en Cuenca - Algoritmo A*")
st.markdown("Encuentra la ruta más corta entre dos puntos turísticos de Cuenca usando el algoritmo A*")

# Inicializar session_state
if 'path' not in st.session_state:
    st.session_state.path = None
if 'distance' not in st.session_state:
    st.session_state.distance = None
if 'explored' not in st.session_state:
    st.session_state.explored = None
if 'start_node' not in st.session_state:
    st.session_state.start_node = None
if 'end_node' not in st.session_state:
    st.session_state.end_node = None

# Crear dos columnas para los controles
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    start_node = st.selectbox("📍 Punto de Inicio", options=sorted(CUENCA_NODES.keys()))

with col2:
    end_node = st.selectbox("🎯 Punto de Destino", options=sorted(CUENCA_NODES.keys()))

with col3:
    st.write("")
    st.write("")
    calculate_button = st.button("🚀 Calcular Ruta", type="primary")

# Ejecutar algoritmo A* cuando se presiona el botón
if calculate_button:
    if start_node == end_node:
        st.warning("⚠️ El punto de inicio y destino son el mismo. Por favor, selecciona ubicaciones diferentes.")
        st.session_state.path = None
    else:
        with st.spinner("Calculando la mejor ruta..."):
            pathfinder = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES, haversine_distance)
            path, distance, explored = pathfinder.find_path(start_node, end_node)
            
            # Guardar en session_state
            st.session_state.path = path
            st.session_state.distance = distance
            st.session_state.explored = explored
            st.session_state.start_node = start_node
            st.session_state.end_node = end_node

# Mostrar resultados si existen en session_state
if st.session_state.path:
    path = st.session_state.path
    distance = st.session_state.distance
    explored = st.session_state.explored
    start_node = st.session_state.start_node
    end_node = st.session_state.end_node
    
    st.success("✅ ¡Ruta encontrada!")
    
    # Mostrar información de la ruta
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("📏 Distancia Total", f"{distance:.2f} km")
    
    with info_col2:
        st.metric("🔢 Nodos en la Ruta", len(path))
    
    with info_col3:
        st.metric("🔍 Nodos Explorados", len(explored))
    
    # Mostrar la ruta con numeración
    st.subheader("🛣️ Ruta Óptima")
    route_steps = " ➜ ".join([f"**{i+1}. {node}**" for i, node in enumerate(path)])
    st.markdown(route_steps)
    
    # Crear mapa con Folium centrado en Cuenca
    st.subheader("🗺️ Visualización del Mapa")
    
    # Centrar en Cuenca
    m = folium.Map(location=[-2.9001, -79.0059], zoom_start=13, tiles="OpenStreetMap")
    
    # Agregar todos los nodos de la ciudad como marcadores grises
    for node_name, coords in CUENCA_NODES.items():
        if node_name not in path:
            folium.Marker(
                location=[coords["lat"], coords["lon"]],
                popup=node_name,
                tooltip=node_name,
                icon=folium.Icon(color="gray", icon="circle", prefix="fa")
            ).add_to(m)
    
    # Agregar marcador de inicio
    folium.Marker(
        location=[CUENCA_NODES[start_node]["lat"], CUENCA_NODES[start_node]["lon"]],
        popup=f"Inicio: {start_node}",
        tooltip=start_node,
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)
    
    # Agregar marcador de destino
    folium.Marker(
        location=[CUENCA_NODES[end_node]["lat"], CUENCA_NODES[end_node]["lon"]],
        popup=f"Destino: {end_node}",
        tooltip=end_node,
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(m)
    
    # Agregar marcadores intermedios de la ruta en amarillo
    for i, node in enumerate(path[1:-1], 1):
        folium.Marker(
            location=[CUENCA_NODES[node]["lat"], CUENCA_NODES[node]["lon"]],
            popup=f"{i}. {node}",
            tooltip=f"Paso {i}: {node}",
            icon=folium.Icon(color="orange", icon="info-sign")
        ).add_to(m)
    
    # Agregar polyline de la ruta con flechas
    path_coords = [[CUENCA_NODES[node]["lat"], CUENCA_NODES[node]["lon"]] for node in path]
    folium.PolyLine(
        path_coords,
        color="blue",
        weight=5,
        opacity=0.8,
        popup="Ruta óptima",
        tooltip="Ruta calculada"
    ).add_to(m)
    
    # Agregar círculos en cada nodo de la ruta para mejor visualización
    for i, node in enumerate(path):
        color = "green" if i == 0 else ("red" if i == len(path) - 1 else "orange")
        folium.CircleMarker(
            location=[CUENCA_NODES[node]["lat"], CUENCA_NODES[node]["lon"]],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=f"Paso {i+1}: {node}"
        ).add_to(m)
    
    # Mostrar el mapa
    st_folium(m, width=1200, height=600, key="route_map")
    
    # Mostrar detalles de cada segmento
    pathfinder = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES, haversine_distance)
    with st.expander("📊 Ver detalles de cada segmento de la ruta"):
        st.write("### Segmentos del recorrido:")
        total = 0
        for i in range(len(path) - 1):
            segment_distance = pathfinder.get_distance(path[i], path[i+1])
            total += segment_distance
            st.write(f"**Segmento {i+1}:** {path[i]} ➜ {path[i+1]} — `{segment_distance:.2f} km`")
        st.write(f"**Total acumulado:** `{total:.2f} km`")
    
    # Mostrar nodos explorados
    with st.expander("🔍 Ver nodos explorados durante la búsqueda"):
        st.write("### Orden de exploración:")
        explored_text = " → ".join([f"{i+1}. {node}" for i, node in enumerate(explored)])
        st.info(explored_text)
        st.write(f"**Total de nodos explorados:** {len(explored)} de {len(CUENCA_NODES)} disponibles")
elif calculate_button and not st.session_state.path:
    st.error("❌ No se encontró una ruta entre los puntos seleccionados.")

# Información adicional
with st.expander("ℹ️ Acerca del Algoritmo A*"):
    st.markdown("""
    **A*** es un algoritmo de búsqueda de caminos que encuentra la ruta más corta entre dos nodos en un grafo.
    
    **Características:**
    - Utiliza una función heurística (distancia haversine) para estimar la distancia al objetivo
    - Garantiza encontrar el camino más corto si existe
    - Es más eficiente que otros algoritmos como Dijkstra cuando se tiene una buena heurística
    
    **Fórmula:** f(n) = g(n) + h(n)
    - g(n): costo real desde el inicio hasta el nodo n
    - h(n): estimación heurística desde n hasta el objetivo
    """)



# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Prueba diferentes combinaciones de inicio y destino para explorar las rutas de Cuenca")