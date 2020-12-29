import networkx as nx
import matplotlib.pyplot as plt
plt.style.use(['science','no-latex'])


g = nx.DiGraph()
nodenames = ['宋丹丹','文章','姚晨','赵薇','陈坤']
g.add_nodes_from(nodenames)
edges = [('宋丹丹','姚晨'),('姚晨','宋丹丹'),
         ('姚晨','赵薇'),('赵薇','姚晨'),
        ('姚晨','陈坤'),('陈坤','姚晨'),
        ('文章','姚晨'),
         ('赵薇','文章'),('文章','赵薇'),
         ('赵薇', '陈坤'), ('陈坤', '赵薇'),
        ('文章', '陈坤'), ('陈坤', '文章')]
g.add_edges_from(edges)
pos = nx.spring_layout(g)
plt.figure(1,figsize=(3.2,3.2), dpi=600)
nx.draw(g, pos, with_labels=True, node_size=1550, node_color="purple", node_shape="o", alpha=0.99, font_family='simsun', font_size=14,font_color="white",font_weight="bold",
		width=1)
plt.savefig("top5-relations.svg", dpi=600)
plt.show()