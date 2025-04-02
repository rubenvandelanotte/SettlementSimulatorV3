import pm4py



# Read the OCEL
ocel = pm4py.read.read_ocel2_json("simulation_log2.jsonocel", encoding= 'utf-8')

# 1. Explore Object Types
object_types = pm4py.ocel_get_object_types(ocel)
print("Object Types:", object_types)


# 2. Analyze Activities per Object Type
activities_per_type = pm4py.ocel_object_type_activities(ocel)
print("Activities per Object Type:", activities_per_type)

# 3. Count Objects per Type per Event
objects_per_event = pm4py.ocel_objects_ot_count(ocel)
header10 = list(objects_per_event.items())[:10]
print("Objects per Event:", header10)  # display a sample

temp_summary = pm4py.ocel_temporal_summary(ocel)
print("Temporal summary: ", temp_summary)

object_summary = pm4py.ocel_objects_summary(ocel)
print("Object Summary: ", object_summary)

interactions_summary = pm4py.ocel_objects_interactions_summary(ocel)
print("Object interactions summary: ", interactions_summary)

# 4. Discover Object-Centric Petri Net (OCPN)
ocpn = pm4py.discover_oc_petri_net(ocel)
pm4py.view_ocpn(ocpn, format='svg')


#5. Object centric directly follows graph
#ocdfg = pm4py.discover_ocdfg(ocel)
#pm4py.view_ocdfg(ocdfg, annotation='frequency', format='svg')

#obj_graph_interaction = pm4py.discover_objects_graph(ocel, graph_type='object_interaction')
#pm4py.view_object_graph(ocel, obj_graph_interaction, format='svg')

#obj_graph_descendants = pm4py.discover_objects_graph(ocel, graph_type='object_descendants')
#pm4py.view_object_graph(ocel, obj_graph_descendants, format='svg')

#obj_graph_inheritance = pm4py.discover_objects_graph(ocel, graph_type='object_inheritance')
#pm4py.view_object_graph(ocel, obj_graph_inheritance, format='svg')


