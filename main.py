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




import pandas as pd
import plotly.graph_objects as go

# Read the data
sankey_data = pd.read_csv('sankey_assignment.csv')

# Prepare the data
df_prep = sankey_data.melt(id_vars=['LABEL'], var_name='source', value_name='value')
df_prep.rename(columns={'LABEL': 'target'}, inplace=True)
df_prep = df_prep[['source', 'target', 'value']]

# Split the data into layers
df_temp1 = df_prep[:40]  # First layer -> Second layer
df_temp2 = df_prep[40:]  # Second layer -> Last layer
df_temp2 = df_temp2[['target', 'source', 'value']]
df_temp2.rename(columns={'target': 'source', 'source': 'target'}, inplace=True)

# Combine the layers
links = pd.concat([df_temp1, df_temp2], axis=0)
unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
mapping_dict = {k: v for v, k in enumerate(unique_source_target)}

# Map the source and target nodes
links['source'] = links['source'].map(mapping_dict)
links['target'] = links['target'].map(mapping_dict)

# Convert to dictionary for Plotly
links_dict = links.to_dict(orient='list')

# Define colors
hex_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2"
]
node_colors = {node: hex_colors[i % len(hex_colors)] for i, node in enumerate(unique_source_target)}

# Plotting
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

# Layout settings
fig.update_layout(title_text="Sankey Diagram with Dynamic Colors", font_size=10)
fig.write_image('sankey_diagram.png')

# Display the diagram
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
