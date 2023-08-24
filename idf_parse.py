'''
parseidf.py


parses an idf file into a dictionary of lists in the following manner:

    each idf object is represented by a list of its fields, with the first
    field being the objects type.

    each such list is appended to a list of objects with the same type in the
    dictionary, indexed by type:

    { [A] => [ [A, x, y, z], [A, a, b, c],
      [B] => [ [B, 1, 2], [B, 1, 2, 3] }

    also, all field values are strings, i.e. no interpretation of the values is
    made.
'''
import ply.lex as lex
import ply.yacc as yacc

from typing import List

import pprint
import json # json.dumps is used for pprinting dictionaries

import networkx as nx
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

# list of token names
tokens = ('VALUE',
          'COMMA',
          'SEMICOLON')

# regular expression rules for simple tokens
t_COMMA = r'[ \t]*,[ \t]*'
t_SEMICOLON = r'[ \t]*;[ \t]*'


# ignore comments, tracking line numbers at the same time
def t_COMMENT(t):
    r'[ \t\r\n]*!.*'
    newlines = [n for n in t.value if n == '\n']
    t.lineno += len(newlines)
    pass
    # No return value. Token discarded


# Define a rule so we can track line numbers
def t_newline(t):
    r'[ \t]*(\r?\n)+'
    t.lexer.lineno += len(t.value)


def t_VALUE(t):
    r'[ \t]*([^!,;\n]|[*])+[ \t]*'
    return t


# Error handling rule
def t_error(t):
    raise SyntaxError("Illegal character '%s' at line %d of input"
                      % (t.value[0], t.lexer.lineno))
    t.lexer.skip(1)


# define grammar of idf objects
def p_idffile(p):
    'idffile : idfobjectlist'
    result = {}
    for idfobject in p[1]:
        name = idfobject[0]
        result.setdefault(name.upper(), []).append(idfobject)
    p[0] = result


def p_idfobjectlist(p):
    'idfobjectlist : idfobject'
    p[0] = [p[1]]


def p_idfobjectlist_multiple(p):
    'idfobjectlist : idfobject idfobjectlist'
    p[0] = [p[1]] + p[2]


def p_idfobject(p):
    'idfobject : objectname SEMICOLON'
    p[0] = [p[1]]


def p_idfobject_with_values(p):
    'idfobject : objectname COMMA valuelist SEMICOLON'
    p[0] = [p[1]] + p[3]


def p_objectname(p):
    'objectname : VALUE'
    p[0] = p[1].strip()


def p_valuelist(p):
    'valuelist : VALUE'
    p[0] = [p[1].strip()]


def p_valuelist_multiple(p):
    'valuelist : VALUE COMMA valuelist'
    p[0] = [p[1].strip()] + p[3]


# oh, and handle errors
def p_error(p):
    raise SyntaxError("Syntax error in input on line %d" % lex.lexer.lineno)


def parse(input) -> dict:
    '''
    parses a string with the contents of the idf file and returns the
    dictionary representation.
    '''
    lexer = lex.lex(debug=False)
    lexer.input(input)
    parser = yacc.yacc()
    result = parser.parse(debug=False)
    return result

class Node:
    def __init__(zone_name, connections=None):
        self.zone_name = zone_name
        self.connections: List[str] = []
        if connections != None:
            assert isinstance(connections, list)
            self.connections = connections

def get_zone_list(parsed_idf):
    ZONE_NAME = 4
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']
    surfaces_set = set()
    for building_surface in building_surfaces:
        surfaces_set.add(building_surface[ZONE_NAME])
    return list(surfaces_set)

def get_surface_to_zone_dict(parsed_idf) -> dict:
    '''
    key: surface_name
    val: zone that the surface is in
    '''
    OUTSIDE_BOUNDARY_CONDITION = 6
    NAME = 1
    ZONE_NAME = 4

    FENESTRATION_SURFACE_NAME = 1
    FENESTRATION_SURFACE_TYPE = 2
    FENESTRATION_SURFACE_BUILDING_SURFACE = 4

    ret_dict = dict()
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']
    fenestration_surfaces = parsed_idf['FENESTRATIONSURFACE:DETAILED']
    print('fen surf:', fenestration_surfaces)
    #print('building_surfaces', building_surfaces)

    # filter ignorable surfaces (eg.Adiabatic)
    for building_surface in building_surfaces:
        # print('ss:', building_surface)
        # print('oof:', building_surface[OUTSIDE_BOUNDARY_CONDITION])
        if building_surface[OUTSIDE_BOUNDARY_CONDITION].upper() not in ['SURFACE', 'ZONE', 'OUTDOORS', 'GROUND']:
            continue
        else:
            ret_dict[building_surface[NAME]] = building_surface[ZONE_NAME]

    for fenestration_surface in fenestration_surfaces:
        if fenestration_surface[FENESTRATION_SURFACE_TYPE] == "Window":
            fen_surface_building_surface = fenestration_surface[FENESTRATION_SURFACE_BUILDING_SURFACE]
            fen_surface_zone = ret_dict[fen_surface_building_surface]
            ret_dict[fenestration_surface[FENESTRATION_SURFACE_NAME]] = fen_surface_zone

    return ret_dict

def get_surface_connect_surface(parsed_idf) -> dict:
    '''
    key: surface name
    val: get zone the surface is connected to
    '''
    OUTSIDE_BOUNDARY_CONDITION = 6
    OUTSIDE_BOUNDARY_CONDITION_OBJECT = 7
    NAME = 1
    ZONE_NAME = 4

    ret_dict = dict()
    building_surfaces = parsed_idf['BUILDINGSURFACE:DETAILED']

    for building_surface in building_surfaces:
        boundary_condition = building_surface[OUTSIDE_BOUNDARY_CONDITION]
        #print('boundary_condition:', boundary_condition)
        if boundary_condition == "Surface":
            ret_dict[building_surface[NAME]] = building_surface[OUTSIDE_BOUNDARY_CONDITION_OBJECT]
        elif boundary_condition == "Ground":
            ret_dict[building_surface[NAME]] = "Ground"
        elif boundary_condition == "Outdoors":
            ret_dict[building_surface[NAME]] = "Outdoors"
        elif boundary_condition == "Zone":
            ret_dict[building_surface[NAME]] = building_surface[OUTSIDE_BOUNDARY_CONDITION_OBJECT]
        else:
            # e.g. adiabatic
            continue

    return ret_dict

def directed_zone_connections(parsed_idf) -> list:
    '''
    @param: parsed_idf - idf file parsed into a dict
    NOTE:
    - outside boundary condition:

    connections of zones. Converting the outputted list will result in a directed graph
    of the zone connection dynamics
    '''
    # below is const for BuildingSurface:Detailed IDF obj
    BUILDING_SURFACE_DETAILED = 0
    NAME = 1
    SURFACE_TYPE = 2
    CONSTRUCTION_NAME = 3
    ZONE_NAME = 4
    SPACE_NAME = 5
    OUTSIDE_BOUNDARY_CONDITION = 6
    OUTSIDE_BOUNDARY_CONDITION_OBJECT = 7
    SUN_EXPOSURE = 8
    WIND_EXPOSURE = 9
    VIEW_FACTOR_TO_GROUND = 10
    NUMBER_OF_VERTICES = 11
    # 12 ~ 23 vertices X,Y,Z stuff

    # get surface_to_zone dict
    surface_to_zone = get_surface_to_zone_dict(parsed_idf)

    # get surface_to_surface connection dict
    surface_connect_surface = get_surface_connect_surface(parsed_idf)

    # get zone_list
    zone_list = get_zone_list(parsed_idf)

    zone_connection_dict = dict()
    zone_connection_list = []
    for surface in surface_to_zone.keys():
        start_zone = surface_to_zone[surface]

        connected_surface = surface_connect_surface[surface]
        # case 1: connected_surface is Ground
        # case 2: connected_surface is Zone name
        # case 3: connecte_surface is Outdoors
        if connected_surface == "Outdoors":
            #zone_connection_dict[start_zone] = 'Outdoors'
            zone_connection_list.append([start_zone, 'Outdoors'])
        elif connected_surface == "Ground":
            #zone_connection_dict[start_zone] = 'Ground'
            zone_connection_list.append([start_zone, 'Ground'])
        elif connected_surface in zone_list:
            #zone_connection_dict[start_zone] = connected_surface # it would be "connected_zone" for this case
            zone_connection_list.append([start_zone, connected_surface])
        else:
            end_zone = surface_to_zone[surface_connect_surface[surface]]
            #zone_connection_dict[start_zone] = end_zone
            zone_connection_list.append([start_zone, end_zone])

    return zone_connection_list

def directed_to_undirected_zone(directed_list):
    '''
    @param: directed_list - list that represents the zone connections as a directed graph
    directed_list: List[List(str, str)]
    '''
    undirected_graph = []
    for connection in directed_list:
        start, end = connection
        undirected_graph.append([start, end])
        undirected_graph.append([end, start])

    return list(undirected_graph)

def main(parsed_idf):
    l = directed_zone_connections(parsed_idf)
    return directed_to_undirected_zone(l)

def visualize_connections(connections):
    G = nx.Graph()
    for connection in connections:
        start, end = connection
        G.add_edge(start, end)
    pos = nx.spring_layout(G, seed=42)  # You can use different layout algorithms if needed
    nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, font_weight="bold")
    plt.show()

def generate_connections(idf_f_path: str):
    idf_file = open(idf_f_path, 'r')
    f = idf_file.read()
    res = parse(f)
    return main(res)

def generate_adjacency(idf_f_path: str):
    edges_list: list = generate_connections(idf_f_path)
    # Create an empty adjacency list dictionary
    adjacency_list = {}

    # Convert the list of edges into an adjacency list dictionary
    for edge in edges_list:
        node1, node2 = edge
        if node1 not in adjacency_list:
            adjacency_list[node1] = []
        if node2 not in adjacency_list:
            adjacency_list[node2] = []

        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)

    for zone in adjacency_list:
        adjacency_list[zone] = list(set(adjacency_list[zone]))

    return adjacency_list

##########################################
#### VARIABLE GENERATION        ##########
##########################################
# for each zone collect variables of:
# - surface outside face incident sky diffuse solar radiation rate
# - humidity
# - temperature

# EplusVariable naming convention:
# 'var-' + <zone_name[:surface_name]> + '-<quantity>'

def get_solar_surface_list(parsed_idf):
    '''
    get surfaces + windows
    '''
    B_SOLAR = 8
    NAME = 1
    ret_list = []
    try:
        for window in parsed_idf['Window']:
            ret_list.append(window[NAME])
    except:
        print("ERROR: provided IDF file doesn't have Window components")

    FENESTRATION_WINDOW_TYPE = 2
    WINDOW_NAME = 1
    try:
        for potential_window in parsed_idf['FENESTRATIONSURFACE:DETAILED']:
            #print('potential window:', potential_window)
            if potential_window[FENESTRATION_WINDOW_TYPE] == 'Window':
                ret_list.append(potential_window[WINDOW_NAME])
    except:
        print("ERROR: provided IDF file doesn't use 'FenestrationSurface:Detailed'")

    for surface in parsed_idf['BUILDINGSURFACE:DETAILED']:
        if surface[B_SOLAR] == 'SunExposed':
            ret_list.append(surface[NAME])
    return ret_list

# write the custom generate_...variable function
# then add the functions
def generate_zone_indoor_temperature(parsed_idf):
    ret = dict()
    zone_list = get_zone_list(parsed_idf)
    for zone in zone_list:
        var_name = 'var-' + str(zone).lower() + '-indoor_temperature'
        ret[var_name] = ("Zone Air Temperature", zone)

    return ret

def generate_zone_relative_humidity(parsed_idf):
    ret = dict()
    zone_list = get_zone_list(parsed_idf)
    for zone in zone_list:
        var_name = 'var-' + str(zone).lower() + '-relative_humidity'
        ret[var_name] =("Zone Air Relative Humidity", str(zone))

    return ret

def generate_zone_sky_diffuse_solar(parsed_idf):
    ret = dict()
    solar_surface_list = get_solar_surface_list(parsed_idf)
    #print('solars surface list:', solar_surface_list)
    surface_to_zone_dict = get_surface_to_zone_dict(parsed_idf)
    #print(surface_to_zone_dict)
    for solar_surface in solar_surface_list:
        var_name = 'var-' + str(solar_surface).lower() + ':' + surface_to_zone_dict[solar_surface] + '-sky_diffuse_solar'
        ret[var_name] = ("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area", solar_surface)

    return ret

def generate_variables(parsed_idf):
    ret = dict()
    # elements of 'generate_variables_list' are functions
    # these are for variables to be collected per zone
    zone_generate_variables_list = [generate_zone_relative_humidity, generate_zone_sky_diffuse_solar, generate_zone_indoor_temperature]
    for generate_function in zone_generate_variables_list:
        temp_dict = generate_function(parsed_idf)
        ret.update(temp_dict)

    # generate settings for zone independent variables
    # provide a list of variable names. the region is autoset as "Environment"
    global_vars = {}
    global_vars_identifier = [
        "Site Outdoor Air Drybulb Temperature",
        "Site Direct Solar Radiation Rate per Area",
        "Site Horizontal Infrared Radiation Rate per Area",
        "Site Outdoor Air Relative Humidity",
    ]
    for global_var in global_vars_identifier:
        var_name = 'var_environment_' + '_'.join(global_var.lower().split())
        global_vars[var_name] = (global_var, "Environment")

    ret.update(global_vars)

    return ret

##########################################
## GNN UTILS
##########################################

def gnn_coo_generate(parsed_idf):
    zone_to_number_dict = gnn_zone_numbering_dict(parsed_idf)

def gnn_zone_to_variables(parsed_idf):
    pass

def gnn_zone_numbering_dict(parsed_idf):
    '''
    Outdoor is always index 0
    Ground is always index 1
    '''
    ret_dict = dict()
    ret_dict['Outdoors'] = 0
    ret_dict['Ground'] = 1

if __name__ == "__main__":
    # idf_file = open('./5ZoneAirCooledConvCoef.idf', 'r')
    idf_file = open('./in.idf', 'r')
    f = idf_file.read()
    res = parse(f)


    print('------------------')
    for i in range(10):
        temp = get_zone_list(res)
        print(temp)
    print('------------------')


    # temp = get_surface_connect_surface(res)
    # temp = generate_variables(res)
    # print(json.dumps(temp, indent=4))
    # print('len:', len(temp))


    l2 = main(res)
    visualize_connections(l2)
    #pp.pprint(l2)

    #print(json.dumps(get_surface_to_zone_dict(res), indent=4))
    # print(res['BUILDINGSURFACE:DETAILED'], type(res['BUILDINGSURFACE:DETAILED']))
    # pp.pprint(get_zone_list(res))
