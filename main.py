import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np




bar_data = pd.read_csv('bar_assignment.csv')
sankey_data = pd.read_csv('sankey_assignment.csv')
network_data = pd.read_csv('networks_assignment.csv')


bar_data = pd.read_csv('bar_assignment.csv')
bar_data['COUNT'] = bar_data['COUNT'].map({1: 'Yes', 0: 'No'})
bar_grouped = bar_data.groupby(['LABEL', 'COUNT']).size().unstack(fill_value=0)


colors = {'Yes': 'blue', 'No': 'red'}
bar_grouped.plot(kind='barh', stacked=True, color=[colors[col] for col in bar_grouped.columns])

plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Horizontal Stacked Bar Chart')
plt.savefig('bar_chart.png')
plt.show()




import pandas as pd
import plotly.graph_objects as go


sankey_data = pd.read_csv('sankey_assignment.csv')


df_prep = sankey_data.melt(id_vars=['LABEL'], var_name='source', value_name='value')
df_prep.rename(columns={'LABEL': 'target'}, inplace=True)
df_prep = df_prep[['source', 'target', 'value']]


df_temp1 = df_prep[:40]  
df_temp2 = df_prep[40:]  
df_temp2 = df_temp2[['target', 'source', 'value']]
df_temp2.rename(columns={'target': 'source', 'source': 'target'}, inplace=True)


links = pd.concat([df_temp1, df_temp2], axis=0)
unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
mapping_dict = {k: v for v, k in enumerate(unique_source_target)}


links['source'] = links['source'].map(mapping_dict)
links['target'] = links['target'].map(mapping_dict)


links_dict = links.to_dict(orient='list')


hex_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2"
]
node_colors = {node: hex_colors[i % len(hex_colors)] for i, node in enumerate(unique_source_target)}


fig = go.Figure(data=[go.Sankey(
   node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=unique_source_target,
        color=[node_colors[node] for node in unique_source_target],
   ),
   link=dict(
        source=links_dict['source'],
        target=links_dict['target'],
        value=links_dict['value'],
        color=[node_colors[unique_source_target[src]] for src in links_dict['source']]
   )
)])


fig.update_layout(title_text="Sankey Diagram with Dynamic Colors", font_size=10)
fig.write_image('sankey_diagram.png')


fig.show()





network_data = pd.read_csv('networks_assignment.csv')


G = nx.Graph()


nodes = network_data.columns[1:].tolist()
G.add_nodes_from(nodes)


for index, row in network_data.iterrows():
    node = row['LABELS']
    for target, value in row[1:].items():
        if value > 0:
            G.add_edge(node, target, weight=value)


pentagon_nodes = ['D', 'F', 'I', 'N', 'S']
angle = np.linspace(0, 2 * np.pi, len(pentagon_nodes), endpoint=False)
pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(pentagon_nodes, angle)}


outer_radius = 2
yellow_nodes = [node for node in nodes if node in [
    'AUT', 'BEL', 'BGR', 'HRV', 'CZE', 'EST', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LUX', 'NLD', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP']]
green_nodes = [node for node in nodes if node in [
    'BIH', 'GEO', 'ISR', 'MNE', 'SRB', 'CHE', 'TUR', 'UKR', 'GBR', 'AUS', 'HKG', 'USA']]
gray_nodes = [node for node in nodes if node in ['ASU']]


outer_nodes = green_nodes + yellow_nodes + gray_nodes


outer_angle = np.linspace(0, 2 * np.pi, len(outer_nodes), endpoint=False)


pos.update({node: (outer_radius * np.cos(a), outer_radius * np.sin(a)) for node, a in zip(outer_nodes, outer_angle)})


color_map = {
    'D': 'blue', 'F': 'blue', 'I': 'blue', 'N': 'blue', 'S': 'blue',
    'BIH': 'green', 'GEO': 'green', 'ISR': 'green', 'MNE': 'green', 'SRB': 'green', 'CHE': 'green', 'TUR': 'green', 'UKR': 'green', 'GBR': 'green', 'AUS': 'green', 'HKG': 'green', 'USA': 'green',
    'AUT': 'yellow', 'BEL': 'yellow', 'BGR': 'yellow', 'HRV': 'yellow', 'CZE': 'yellow', 'EST': 'yellow', 'FRA': 'yellow', 'DEU': 'yellow', 'GRC': 'yellow', 'HUN': 'yellow', 'IRL': 'yellow', 'ITA': 'yellow', 'LVA': 'yellow', 'LUX': 'yellow', 'NLD': 'yellow', 'PRT': 'yellow', 'ROU': 'yellow', 'SVK': 'yellow', 'SVN': 'yellow', 'ESP': 'yellow',
    'ASU': 'gray'
}
node_colors = [color_map.get(node, 'gray') for node in G.nodes()]


display_nodes = pentagon_nodes + outer_nodes
display_edges = [(u, v) for u, v, d in G.edges(data=True) if u in display_nodes and v in display_nodes]


H = G.edge_subgraph(display_edges).copy()


subgraph_pos = {node: pos[node] for node in H.nodes()}
subgraph_colors = [color_map[node] for node in H.nodes()]


edge_colors = []
for u, v in H.edges():
    if v in yellow_nodes:
        edge_colors.append('yellow')
    elif v in green_nodes:
        edge_colors.append('green')
    elif v in gray_nodes:
        edge_colors.append('gray')
    else:
        edge_colors.append('blue')


plt.figure(figsize=(10, 8))
nx.draw(H, subgraph_pos, with_labels=True, node_color=subgraph_colors, node_size=800, font_size=8, font_weight='bold', edge_color=edge_colors, width=2)


edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in H.edges()}
nx.draw_networkx_edge_labels(H, subgraph_pos, edge_labels=edge_labels, font_size=6)

plt.title("Network Graph", fontweight='bold')
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
