import graphviz

g = graphviz.Digraph('unix', filename='ID3.gv')
def buildg(t,par,C,edge_label):
    if len(t)==1:
        C[0]+=1
        g.attr('node',shape='circle',style='filled',color='green')
        g.node(str(C[0]),label=t['node name'])
        if par!=-1:
            g.edge(str(par),str(C[0]),label=edge_label)
        return
    C[0]+=1
    v=C[0]
    g.attr('node',shape='box',style='filled',color='lightblue2')
    g.node(str(v),label=t['node name'])
    if par!=-1:
        g.edge(str(par),str(v),label=edge_label)
    for s in t:
        if s!='node name':
            buildg(t[s],v,C,s)
    
def show(t):
    C=[0]
    buildg(t,-1,C,'')
    g.view()
    
#d={
#    "node name": "Outlook",
#    "Sunny": {
#        "node name": "Humidity",
#        "High": {
#            "node name": "No"
#        },
#        "Normal": {
#            "node name": "Yes"
#        }
#    },
#    "Overcast": {
#        "node name": "Yes"
#    },
#    "Rain": {
#        "node name": "Wind",
#        "Weak": {
#            "node name": "Yes"
#        },
#        "Strong": {
#            "node name": "No"
#        }
#    }
#}
#    
#show(d)