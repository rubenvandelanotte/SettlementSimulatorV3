import pm4py



# Read the OCEL
ocel = pm4py.read.read_ocel2_json("sample_ocel_log.jsonocel", encoding= 'utf-8')

# Get the list of object types in the OCEL
object_types = pm4py.ocel_get_object_types(ocel)
print("Object types: ", object_types)

# Return an OCEL with a subset of randomly chosen objects
sampled_ocel = pm4py.sample_ocel_objects(ocel, 50)

# Get the list of object types in the sampled OCEL
object_types_sampled = pm4py.ocel_get_object_types(sampled_ocel)
print("Object types in the sampled OCEL: ", object_types_sampled)

# Get the set of activities for each object type
ot_activities_sampled = pm4py.ocel_object_type_activities(sampled_ocel)
print("Activities per object types in the sampled OCEL: ", ot_activities_sampled)

# Count for each event the number of objects per type
objects_ot_count_sampled = pm4py.ocel_objects_ot_count(sampled_ocel)
print("Number of related objects per type in the sampled OCEL: ", objects_ot_count_sampled)
ocpn = pm4py.discover_oc_petri_net(ocel)
pm4py.view_ocpn(ocpn, format='svg')
