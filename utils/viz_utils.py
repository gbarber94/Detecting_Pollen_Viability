import numpy as np
from pandas.io.json import json_normalize
import pandas as pd 
import itertools
import plotly.graph_objects as go 
from plotly.offline import iplot
from PIL import Image
from collections import defaultdict


def outputdict_to_df(output_dict,image_path,score_threshold,image_cords_flip = True):
    """
    Takes in output_dict, filters by dectection_score threshold, converts bounding box cords to pixels
    and outputs dataframe with columns ['detection_classes', 'detection_scores', 'ymin', 'xmin', 'ymax','xmax']
    """
    
    df  = json_normalize(output_dict)
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    #build column detection classes
    dc = df['detection_classes'].apply(pd.Series).T
    dc.columns = ['detection_classes'] 
    
    #build column detection scores
    ds = df['detection_scores'].apply(pd.Series).T
    ds.columns = ['detection_scores'] 
    
    #build df boundry box cordinates
    db = pd.DataFrame(df['detection_boxes'][0])
    db.columns = ['ymin', 'xmin', 'ymax', 'xmax']
    
    #convert cordinates to pixle loactions
    if image_cords_flip == True: #check if image is flipped vertically
        db[['ymin','ymax']] = (1-db[['ymin','ymax']])*img_height
        db[['xmin','xmax']] = db[['xmin','xmax']]*img_width
    else: 
        db[['ymin','ymax']] = (db[['ymin','ymax']])*img_height
        db[['xmin','xmax']] = db[['xmin','xmax']]*img_width
    
    #build df
    df = pd.concat([dc,ds,db], axis = 1)
    #threshold subset
    df = df[df['detection_scores'] > score_threshold]
    
    return df

def row_to_xml(xml,row):
    """
    Takes in row of outputdict df and converts to xml
    """
    xml.append('    <object>')
    xml.append('        <name>{}</name>'.format(row.loc['detection_classes']))
    xml.append('        <pose>Unspecified</pose>')
    xml.append('        <truncated>0</truncated>')
    xml.append('        <difficult>0</difficult>')
    xml.append('        <bndbox>')
    xml.append('            <xmin>{}</xmin>'.format(row.loc['xmin'].astype(int)))
    xml.append('            <ymin>{}</ymin>'.format(row.loc['ymin'].astype(int)))
    xml.append('            <xmax>{}</xmax>'.format(row.loc['xmax'].astype(int)))
    xml.append('            <ymax>{}</ymax>'.format(row.loc['ymax'].astype(int)))
    xml.append('        </bndbox>')
    xml.append('    </object>')
    return xml

def outputdict_to_xml(output_dict,image_path):
    """
    Takes in outputdict and outputs xml file of same format produced by LabelImg.
    """
    
    df = outputdict_to_df(output_dict,image_path,score_threshold = 0.5,image_cords_flip = False) #LabelImg flip cordinates
    file_name = image_path.split("\\")[1]
    xml = ['<annotation>','    <filename>{}</filename>'.format(file_name)]
    df.apply(lambda x : row_to_xml(xml,x), axis = 1)
    xml.append('</annotation>')
    xml = '\n'.join(xml)
    file_name = file_name.split(".")[0]
    with open("{}.xml".format(file_name), "w") as f:
        f.write(xml)
    return f'xml out: {file_name}.xml'
    

def add_bbox_colors(df):
    """
    Adds column with unique color value for each detection class up to 10 classes. For greater then 10 classes additonal colors
    may be append to box_colors.
    """
    
    box_colors = [
    '#ff7f0e',  # safety orange
    '#1f77b4',  # muted blue
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]
    # List of classes
    classes = np.sort(df['detection_classes'].unique())
    class_number = len(classes)
    if class_number <= len(box_colors):
        box_colors = box_colors[:class_number]
        # {class:color} dictionary
        box_colors = dict(zip(classes, box_colors))
        df = df.sort_values(by=['detection_classes'])
        df['box_color'] = df['detection_classes'].map(box_colors)
        return df
    else:
        return "detection class number exceedes box pallette see boundry_box_colors()"

def get_bbox_shape(row,scale_factor):
    """
    Takes in row of outputdict df, and formats boundry box as plotly shape object.
    """
    
    shape_list = [{'line': {'color': row.loc['box_color']},
               'type': 'rect',
                 'x0': row.loc['xmin']*scale_factor,
                 'x1': row.loc['xmax']*scale_factor,
                 'y0': row.loc['ymin']*scale_factor,
                 'y1': row.loc['ymax']*scale_factor}]
    return shape_list

def get_bbox_button(row,scale_factor):
    """
    Takes in row of outputdict df and returns on/off button for the boundry box.
    """
    
    button_list = [{'args': ['shapes',[{'line': {'color': row.loc['box_color']},
                   'type': 'rect',
                     'x0': row.loc['xmin']*scale_factor,
                     'x1': row.loc['xmax']*scale_factor,
                     'y0': row.loc['ymin']*scale_factor,
                     'y1': row.loc['ymax']*scale_factor}]],
                   'label': row.loc['detection_classes'],
                  'method': 'relayout'}]
    return button_list

def bbox_plot(df,image_path,scale_factor=1, save = False,display = True):
    """
    Takes in df constructed from output_dict, path to image and scaling factor for image.
    Outputs plotly plot of boundry boxes overlaying the input image
    """
    
    image = Image.open(image_path)
    img_width, img_height = image.size
    df = add_bbox_colors(df)
    bbox_list = df.apply(lambda x : get_bbox_shape(x,scale_factor), axis = 1)
    bbox_list = list(itertools.chain.from_iterable(bbox_list))
    plot = go.Figure({'data': [], 
           'layout': {
               'shapes': bbox_list, 
               'images': [{'layer': 'below',
                           'opacity': 1.0,
                           'sizex': img_width*scale_factor,
                           'sizey': img_height*scale_factor,
                          # 'sizing': 'stretch',
                           'source': image,
                           'x': 0,
                           'xref': 'x',
                           'y': img_height*scale_factor,
                           'yref': 'y'}],
               'margin': {'b': 0, 'l': 0, 'r': 0, 't': 0,'pad':4},
               'width': img_width*scale_factor,
               'height':img_height*scale_factor,
               'xaxis': {'range': [0, img_width*scale_factor], 'visible': False},
               'yaxis': {'range': [0, img_height*scale_factor], 'scaleanchor': 'x', 'visible': False},
               'plot_bgcolor':'rgba(0,0,0,0)'
              }
          })
    if save == True:
        img_name = image_path.split('\\')[1].split('.')[0]
        plot.write_image(f"results/{img_name}_bbox.jpg")
        if display == False:
            return
        
    config = {'displaylogo': False,'displayModeBar': True, 'modeBarButtonsToRemove':['zoomIn2d','zoomOut2d','pan2d','autoScale2d']} 

    return iplot(plot,config = config)

def twoclass_bbox_plot(df,image_path,scale_factor=1):
    """
    For two class adds toggleable buttons for boundry boxes to plot
    """
    
    image = Image.open(image_path)
    img_width, img_height = image.size
    df = add_bbox_colors(df)
    bbox_list = df.apply(lambda x : get_bbox_shape(x,scale_factor), axis = 1)
    bbox_list = list(itertools.chain.from_iterable(bbox_list))
    
    button_dict = df.apply(lambda x: get_bbox_button(x,scale_factor), axis = 1)
    button_dict = list(itertools.chain.from_iterable(button_dict))
    button_dict = sorted(button_dict, key = lambda i: i['label']) # sort button dictionaries

    button_list = defaultdict(list)
    for i in button_dict:     
        key = i['label']
        button_list[key].append(i)

    if len(button_list) < 2:
        button_list.append([])
        
    #Entire Selection Buttons
    All = {'args': ['shapes', bbox_list],'label': 'All', 'method': 'relayout'}
    none = {'args': ['shapes', []],'label': 'None', 'method': 'relayout'}

    button_list[1].insert(0,All)
    button_list[2].insert(0,none)
    
    plot = go.Figure({'data': [], 
           'layout': {
               'shapes': bbox_list, 
               'images': [{'layer': 'below',
                           'opacity': 1.0,
                           'sizex': img_width*scale_factor,
                           'sizey': img_height*scale_factor,
                          # 'sizing': 'stretch',
                           'source': image,
                           'x': 0,
                           'xref': 'x',
                           'y': img_height*scale_factor,
                           'yref': 'y'}],
               'margin': {'b': 0, 'l': 0, 'r': 0, 't': 0,'pad':4},
               'width': img_width*scale_factor,
               'height':img_height*scale_factor,
               'xaxis': {'range': [0, img_width*scale_factor], 'visible': False},
               'yaxis': {'range': [0, img_height*scale_factor], 'scaleanchor': 'x', 'visible': False},
               'plot_bgcolor':'rgba(0,0,0,0)',
               'updatemenus': [{'buttons': button_list[2],
                                'type': 'buttons','pad':{"r": 1}}, {'buttons': button_list[1],
                                'type': 'buttons', 'pad':{"r": 60}}],
               'plot_bgcolor':'rgba(0,0,0,0)'
              }
          })
    config = {'displaylogo': False,'displayModeBar': True, 'modeBarButtonsToRemove':['zoomIn2d','zoomOut2d','pan2d','autoScale2d']} 

    return iplot(plot,config = config)

