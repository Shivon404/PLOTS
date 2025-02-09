import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image



bar_data = pd.read_csv('bar_assignment.csv')
sankey_data = pd.read_csv('sankey_assignment.csv')
network_data = pd.read_csv('networks_assignment.csv')


bar_data['COUNT'] = bar_data['COUNT'].map({1: 'Yes', 0: 'No'})


bar_grouped = bar_data.groupby(['LABEL', 'COUNT']).size().unstack(fill_value=0)


bar_grouped.plot(kind='barh', stacked=True)
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Horizontal Stacked Bar Chart')
plt.savefig('bar_chart.png')
plt.show()





sankey_data = pd.read_csv('sankey_assignment.csv')  


labels = ['PS', 'OMP', 'CNP', 'NRP', 'NMCCC', 'PEC', 'NCDM', 'RGS', 'Reg', 'Aca', 'Oth']
source = [0, 1, 2, 3, 4, 5, 6, 7]
target = [8, 9, 10, 8, 9, 10, 8, 9]
value = sankey_data.iloc[0, 1:9].tolist()


node_colors = [
    "rgba(31, 119, 180, 0.8)",  
    "rgba(255, 127, 14, 0.8)",  
    "rgba(44, 160, 44, 0.8)",   
    "rgba(214, 39, 40, 0.8)",   
    "rgba(148, 103, 189, 0.8)", 
    "rgba(140, 86, 75, 0.8)",   
    "rgba(227, 119, 194, 0.8)", 
    "rgba(127, 127, 127, 0.8)"  
]


link_colors = [node_colors[src] for src in source]


fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=labels,
        color=node_colors + ["rgba(255, 182, 193, 0.8)", "rgba(255, 222, 173, 0.8)", "rgba(192, 192, 192, 0.8)"]  
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
)])


fig.update_layout(title_text="Sankey Diagram", font_size=10)
fig.write_image("sankey_diagram.png")
fig.show()





network_data = pd.read_csv('networks_assignment.csv')


G = nx.Graph()


colors = {
    'D': 'blue', 'F': 'blue', 'I': 'blue', 'N': 'blue', 'S': 'blue',  # Pentagram nodes
    'BIH': 'green', 'GEO': 'green', 'ISR': 'green', 'MNE': 'green', 'SRB': 'green',
    'CHE': 'green', 'TUR': 'green', 'UKR': 'green', 'GBR': 'green', 'AUS': 'green',
    'HKG': 'green', 'USA': 'green',
    'AUT': 'yellow', 'BEL': 'yellow', 'BGR': 'yellow', 'HRV': 'yellow', 'CZE': 'yellow',
    'EST': 'yellow', 'FRA': 'yellow', 'DEU': 'yellow', 'GRC': 'yellow', 'HUN': 'yellow',
    'IRL': 'yellow', 'ITA': 'yellow', 'LVA': 'yellow', 'LUX': 'yellow', 'NLD': 'yellow',
    'PRT': 'yellow', 'ROU': 'yellow', 'SVK': 'yellow', 'SVN': 'yellow', 'ESP': 'yellow',
    'ASU': 'green'  
}



for node, color in colors.items():
    G.add_node(node, color=color)


for index, row in network_data.iterrows():
    label = row['LABELS']
    for col in network_data.columns[1:]:
        if row[col] > 0:
            G.add_edge(label, col, weight=row[col])


pos = {
    'D': (0, 0.5), 'F': (-0.5, -0.3), 'I': (0.5, -0.3), 'N': (-0.3, -0.8), 'S': (0.3, -0.8),
    'BIH': (-1.5, 1), 'GEO': (-1.0, 1.5), 'ISR': (-0.5, 1.8), 'MNE': (0, 2),
    'SRB': (0.5, 1.8), 'CHE': (1.0, 1.5), 'TUR': (1.5, 1), 'UKR': (2, 0.5),
    'GBR': (2.5, 0), 'AUS': (2, -0.5), 'HKG': (1.5, -1), 'USA': (1, -1.5),
    'AUT': (-1.5, -1), 'BEL': (-1.0, -1.5), 'BGR': (-0.5, -1.8), 'HRV': (0, -2),
    'CZE': (0.5, -1.8), 'EST': (1.0, -1.5), 'FRA': (1.5, -1), 'DEU': (-2, 0.5),
    'GRC': (-2.5, 0), 'HUN': (-2, -0.5), 'IRL': (-1.5, -1), 'ITA': (-1, -1.5),
    'LVA': (-0.5, -1.8), 'LUX': (0, -2), 'NLD': (0.5, -1.8), 'PRT': (1, -1.5),
    'ROU': (1.5, -1), 'SVK': (2, -0.5), 'SVN': (2.5, 0), 'ESP': (-2.5, 0),
    'ASU': (2, 2)  
}



node_colors = [colors[node] for node in G.nodes()]


edge_weights = [G[u][v]['weight'] for u, v in G.edges()]


plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, node_color=node_colors, with_labels=True, edge_color='gray', 
    node_size=800, font_size=8, font_weight='bold', width=edge_weights
)


edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title("Network Graph with Pentagram Center")
plt.savefig('network_graph.png')
plt.show()




bar_img = Image.open("bar_chart.png")  
sankey_img = Image.open("sankey_diagram.png")
network_img = Image.open("network_graph.png")


network_width = int(network_img.width * 1.5)  
network_height = int(network_img.height * 1.5)
network_img = network_img.resize((network_width, network_height))

bar_width = network_width // 3
bar_height = network_height // 3
bar_img = bar_img.resize((bar_width, bar_height))

sankey_width = bar_width
sankey_height = network_height - bar_height  
sankey_img = sankey_img.resize((sankey_width, sankey_height))


collated_width = network_width + bar_width
collated_height = network_height
collated_img = Image.new('RGB', (collated_width, collated_height), "white")


collated_img.paste(bar_img, (0, 0))
collated_img.paste(sankey_img, (0, bar_img.height))  
collated_img.paste(network_img, (bar_width, 0))  

collated_img.save("collated_graphs.png")
collated_img.show()
