from xml.dom import minidom
import os
from lxml import etree

# information that we need / would get from DQN arrangement

# Embedded
sections = ['base','shoulder','elbow','wrist1','wrist2','wrist3'] # make this work for infinitely long arms?

# namespace information
XHTML_NAMESPACE = "http://www.ros.org/wiki/xacro"
XHTML = "{%s}" % XHTML_NAMESPACE # substituting in namespace prefix

NSMAP = {'xacro' : XHTML_NAMESPACE} # defining prefix, uri for namespace

def input_parser(arrangement, info):
    order = [''] * len(arrangement)
    act_types = []
    brac_types = []
    link_types = []
    grip_type = ''
    print('arrangement: ', arrangement)
    for i in range(len(arrangement)):
        if arrangement[i] == '':
            continue
        order[i] = arrangement[i][0]
        if arrangement[i][0] == 'a':
            act_types.append(arrangement[i][1])
        elif arrangement[i][0] == 'b':
            brac_types.append(arrangement[i][1])
        elif arrangement[i][0] == 'l':
            link_types.append(arrangement[i][1:len(arrangement[i])])
        else:
            grip_type = arrangement[i][1]
    robot_name = info[0]
    robot_version = info[1]
    #print('order: ', order)
    return order, act_types, brac_types, link_types, grip_type, robot_name, robot_version

def actuator(root, a_cnt, xhtml, type, names, elem_cnt):
    etree.SubElement(root,xhtml + 'actuator',type=type[a_cnt],name=names[elem_cnt],child=names[elem_cnt + 1])
    return a_cnt + 1

def bracket(root, b_cnt, xhtml, types, names, elem_cnt):
    etree.SubElement(root,xhtml + 'bracket',type=types[b_cnt],name=names[elem_cnt],child=names[elem_cnt + 1])
    return b_cnt + 1

def link(root, l_cnt, xhtml, link_types, names, elem_cnt):
    etree.SubElement(root,xhtml + 'link',type=link_types[l_cnt][0],
                     extension=link_types[l_cnt][1],twist=link_types[l_cnt][2],name=names[elem_cnt],child=names[elem_cnt + 1])
    return l_cnt + 1

def gripper(root, xhtml, grip_type):
    etree.SubElement(root, xhtml + 'gripper', type=grip_type, name="end_effector", mass="0.0")

def get_names(order):
    names = [''] * len(order)
    elem_cnt = len(order) - 1
    a_cnt = order.count('a') + 1
    for elem in reversed(order):
        if elem == 'g':
            names[elem_cnt] = "end_effector"
            a_cnt -= 1
        elif elem == 'a':
            if a_cnt - 1 >= len(sections):
                names[elem_cnt] = "Arm/J" + str(a_cnt) + "_wrist" + str(a_cnt - 3)
            else:
                names[elem_cnt] = "Arm/J" + str(a_cnt) + "_" + sections[a_cnt - 1]
        elif elem == 'b':
            if a_cnt - 1 >= len(sections):
                names[elem_cnt] = "wrist" + str(a_cnt - 3) + "_bracket"
            else:
                names[elem_cnt] = sections[a_cnt - 1] + "_bracket"
            a_cnt -= 1
        elif elem == 'l':
            if a_cnt - 1 >= len(sections):
                names[elem_cnt] = "wrist" + str(a_cnt - 4) + "_wrist" + str(a_cnt - 3)
            else:
                names[elem_cnt] = sections[a_cnt - 2] + "_" + sections[a_cnt - 1]
            a_cnt -= 1
        elem_cnt -= 1
    #print(names)
    return names

def make_xml(arrangement, info):
    order, act_types, brac_types, link_types, grip_type, robot_name, robot_version = input_parser(arrangement,info)
    names = get_names(order)
    a_cnt = 0
    b_cnt = 0
    l_cnt = 0
    elem_cnt = 0
    root = etree.Element("robot", nsmap=NSMAP, version=robot_version,name=robot_name)
    tree = etree.ElementTree(root)
    comment = etree.Comment(' HEBI ' + robot_name + ' Arm Kit ')
    root.insert(1,comment)
    ## WILL NOT CHANGE FROM HERE
    include = etree.SubElement(root, XHTML + "include",filename='$(find hebi_description)/urdf/hebi.xacro')
    arg = etree.SubElement(root,XHTML + 'arg',name='hebi_base_frame',default='world')
    property = etree.SubElement(root,XHTML + 'property',name='hebi_base_frame',value='$(arg hebi_base_frame)')
    _if = etree.SubElement(root,XHTML + 'if',value="${hebi_base_frame == 'world'}")
    if_link = etree.SubElement(_if,'link', name="$(arg hebi_base_frame)")
    joint = etree.SubElement(root,'joint',name="$(arg hebi_base_frame)_joint", type="fixed")
    origin = etree.SubElement(joint,'origin',xyz="0 0 0", rpy="0 0 0")
    parent = etree.SubElement(joint,'parent',link='$(arg hebi_base_frame)')
    child = etree.SubElement(joint,'child',link="Arm/J1_base/INPUT_INTERFACE")
    ## TO HERE
    for elem in order:
        if elem == 'g':
            gripper(root, XHTML, grip_type)
        elif elem == 'a':
            a_cnt = actuator(root, a_cnt, XHTML, act_types, names, elem_cnt)
        elif elem == 'b':
            b_cnt = bracket(root, b_cnt, XHTML, brac_types, names, elem_cnt)
        elif elem == 'l':
            l_cnt = link(root, l_cnt, XHTML, link_types, names, elem_cnt)
        elem_cnt += 1
    tree.write("dqn_sac.xacro",encoding='UTF-8', xml_declaration=True, pretty_print=True)