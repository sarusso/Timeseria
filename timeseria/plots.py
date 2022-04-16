# -*- coding: utf-8 -*-
"""Plotting utilities."""
# Note: the code in this module is spaghetti-ish code, and there are no tests. A major refactoring is required.'''

import os
import uuid
import datetime
from .time import dt_from_s, dt_to_str, dt_from_str
from .datastructures import DataTimePointSeries, DataTimeSlotSeries
from .units import TimeUnit
from .utilities import is_numerical, os_shell
try:
    from pyppeteer.chromium_downloader import download_chromium,chromium_executable
    image_plot_support=True  
except ImportError:
    image_plot_support=False            

# Setup logging
import logging
logger = logging.getLogger(__name__)

# Setup come configuration
AGGREGATE_THRESHOLD = int(os.environ.get('AGGREGATE_THRESHOLD', 10000))
RENDER_MARK_AS_INDEX = False
DEFAULT_PLOT_TYPE = os.environ.get('DEFAULT_PLOT_TYPE', None)
if DEFAULT_PLOT_TYPE:
    if DEFAULT_PLOT_TYPE == 'interactive':
        logger.debug('Setting default plot type to "interactive"')
        DEFAULT_PLOT_AS_IMAGE = False
    elif DEFAULT_PLOT_TYPE == 'image':
        logger.debug('Setting default plot type to "image"')
        DEFAULT_PLOT_AS_IMAGE = True         
    else:
        raise ValueError('Unknown plot type "{}" for DEFAULT_PLOT_TYPE'.forat(DEFAULT_PLOT_TYPE))
else:
    DEFAULT_PLOT_AS_IMAGE=False

#=================
#   Utilities
#=================

# Tab 10 colormap (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
# https://github.com/matplotlib/matplotlib/blob/f6e0ee49c598f59c6e6cf4eefe473e4dc634a58a/lib/matplotlib/_cm.py
_tab10_colormap_norm = (
    (0.12156862745098039, 0.4666666666666667,  0.7058823529411765  ),  # 1f77b4
    #(1.0,                 0.4980392156862745,  0.054901960784313725),  # ff7f0e
    (0.17254901960784313, 0.6274509803921569,  0.17254901960784313 ),  # 2ca02c
    (0.8392156862745098,  0.15294117647058825, 0.1568627450980392  ),  # d62728
    (0.5803921568627451,  0.403921568627451,   0.7411764705882353  ),  # 9467bd
    (0.5490196078431373,  0.33725490196078434, 0.29411764705882354 ),  # 8c564b
    #(0.8901960784313725,  0.4666666666666667,  0.7607843137254902  ),  # e377c2
    (0.4980392156862745,  0.4980392156862745,  0.4980392156862745  ),  # 7f7f7f
    (0.7372549019607844,  0.7411764705882353,  0.13333333333333333 ),  # bcbd22
    (0.09019607843137255, 0.7450980392156863,  0.8117647058823529),    # 17becf
)

def to_rgba_str_from_norm_rgb(rgb, a):
    return 'rgba({},{},{},{})'.format(rgb[0]*255,rgb[1]*255,rgb[2]*255,a)

def _utc_fake_s_from_dt(dt):
    dt_str = dt_to_str(dt)
    if '+' in dt_str:
        return s_from_dt(dt_from_str(dt_str.split('+')[0]+'+00:00'))
    elif '-' in dt_str:
        return s_from_dt(dt_from_str(dt_str.split('-')[0]+'+00:00'))
    else:
        raise ValueError('Cannot convert to fake UTC epoch a datetime without a timezone')



def _check_data_for_plot(data):
    if is_numerical(data):
        keys = []
    elif isinstance(data, list):
        keys = []
        for key,item in enumerate(data): 
            if not is_numerical(item):
                raise Exception('Don\'t know how to plot item "{}" of type "{}"'.format(item, item.__class__.__name__))
            keys.append(key)
    
    elif isinstance(data, dict):
        keys = []
        for key,item in data.items(): 
            if not is_numerical(item):
                raise Exception('Don\'t know how to plot item "{}" of type "{}"'.format(item, item.__class__.__name__))
            keys.append(key)            
    else:
        raise Exception('Don\'t know how to plot data "{}" of type "{}"'.format(data, data.__class__.__name__))
    return keys

from timeseria.time import s_from_dt
def _to_dg_time(dt):
    '''Get Dygraphs time form datetime'''
    #return '{}{:02d}-{:02d}T{:02d}:{:02d}:{:02d}+00:00'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute,dt.second)
    #return s_from_dt(dt)*1000
    if dt.microsecond:
        return 'new Date(Date.UTC({}, {}, {}, {}, {}, {}, {}))'.format(dt.year, dt.month-1, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)
    else:
        return 'new Date(Date.UTC({}, {}, {}, {}, {}, {}))'.format(dt.year, dt.month-1, dt.day, dt.hour, dt.minute, dt.second)


def _to_dg_data(serie, data_labels_to_plot, data_indexes_to_plot, full_precision, aggregate_by=0):
    '''{% for timestamp,value in metric_data.items %}{{timestamp}},{{value}}\n{%endfor%}}'''
    
    dg_data=''
    first_t  = None
    last_t   = None

    global_min = None
    global_max = None
    
    if serie.mark:
        serie_mark_start_t = s_from_dt(serie.mark[0])
        serie_mark_end_t = s_from_dt(serie.mark[1])

    for i, item in enumerate(serie):
        
        # Offset
        i=i+1
                
        # Prepare a dict for every piece of data
        try:
            labels
        except UnboundLocalError:
            # Set labels and support vars
            labels = data_labels_to_plot
            if not labels:
                labels = [None]
            data_sums = [0 for label in labels]
            data_mins = [None for label in labels]
            data_maxs = [None for label in labels]
            index_sums = [0 for index in data_indexes_to_plot]
            

        #====================
        #  Aggregation
        #====================        
        if aggregate_by:

            # Set first timestamp
            if first_t is None:
                first_t = item.t
                  
            # Define data
            if labels:
                datas=[]      
                for j, label in enumerate(labels):
                    if label is None:
                        datas.append(item.data)
                    else:
                        datas.append(item.data[label])
            else:
                datas = [item.data]
            
            
            # Loop over data labels, including the "None" label if we have no labels (float data), and add data
            for j, data in enumerate(datas):
                
                # Sum
                data_sums[j] += data
                
                # Global min
                if global_min is None:
                    global_min = data
                else:
                    if data < global_min:
                        global_min = data                
                
                # Global max
                if global_max is None:
                    global_max = data
                else:
                    if data > global_max:
                        global_max = data

                # Min
                if data_mins[j] is None:
                    data_mins[j] = data
                else:
                    if data < data_mins[j]:
                        data_mins[j] = data
    
                # Max
                if data_maxs[j] is None:
                    data_maxs[j] = data
                else:
                    if data > data_maxs[j]:
                        data_maxs[j] = data

            # Loop over series data_indexes and add data
            for j, index in enumerate(data_indexes_to_plot):
                try:
                    index_value = item.data_indexes[index]
                    if index_value is not None:
                        index_sums[j] += index_value
                except KeyError:
                    pass

            # Dump aggregated data?
            if (i!=0)  and ((i % aggregate_by)==0):
                if isinstance(serie, DataTimePointSeries):
                    last_t = item.t
                elif isinstance(serie, DataTimeSlotSeries):
                    last_t = item.end.t
                aggregation_t = first_t + ((last_t-first_t) /2)
                data_part=''
                
                # Data
                for i, label in enumerate(labels):
                    avg = data_sums[i]/aggregate_by
                    if full_precision:
                        data_part+='[{},{},{}],'.format(data_mins[i], avg, data_maxs[i])
                    else:
                        data_part+='[{:.2f},{:.2f},{:.2f}],'.format(data_mins[i], avg, data_maxs[i])
                        
                # data_indexes
                for i, index in enumerate(data_indexes_to_plot):
                    if index_sums[i] is not None:
                        aggregated_index_value = index_sums[i]/aggregate_by
                        if full_precision:
                            data_part+='[0,{},{}],'.format(aggregated_index_value,aggregated_index_value)
                        else:
                            data_part+='[0,{:.4f},{:.4f}],'.format(aggregated_index_value,aggregated_index_value)                            
                    else:
                        data_part+='[null,null,null],'

                # Do we have a mark?
                if serie.mark and RENDER_MARK_AS_INDEX:
                    if item.start.t >= serie_mark_start_t and item.end.t < serie_mark_end_t:                                    
                        # Add the (active) mark
                        data_part+='[0,1,1],'
                    else:
                        # Add the (inactive) mark
                        data_part+='[0,0,0],'
                
                # Remove last comma
                data_part = data_part[0:-1]
                
                # Add to dg_data
                dg_data += '[{},{}],'.format(_to_dg_time(dt_from_s(aggregation_t, tz=item.tz)), data_part)
                
                # Reset averages
                data_sums = [0 for label in labels]
                data_mins = [None for label in labels]
                data_maxs = [None for label in labels]
                index_sums = [0 for index in data_indexes_to_plot]
                first_t = None

        #====================
        #  No aggregation
        #====================
        else:
                    
            # Loop over data labels, including the "None" label if we have no labels (float data), and add data
            data_part=''
            for label in labels:
                if label is None:
                    data = item.data
                else:
                    data = item.data[label]

                if full_precision:
                    data_part += '{},'.format(data)
                else:
                    data_part += '{:.2f},'.format(data)
                                    
                # Global min
                if global_min is None:
                    global_min = data
                else:
                    if data < global_min:
                        global_min = data                
                
                # Global max
                if global_max is None:
                    global_max = data
                else:
                    if data > global_max:
                        global_max = data
                
            # Loop over series data_indexes and add data
            for index in data_indexes_to_plot:
                try:
                    index_value = item.data_indexes[index]
                    if index_value is not None:
                        if full_precision:
                            data_part+='{},'.format(index_value)
                        else:
                            data_part+='{:.4f},'.format(index_value)                       
                    else:
                        data_part+='null,'
                except KeyError:
                    data_part+='null,'

            # Do we have a mark?
            if serie.mark and RENDER_MARK_AS_INDEX:
                if item.t >= serie_mark_start_t and item.t < serie_mark_end_t:
                    # Add the (active) mark
                    data_part+='1,'
                else:
                    # Add the (inactive) mark
                    data_part+='0,'

            # Remove last comma
            data_part = data_part[0:-1]

            # Add to dg_data
            dg_data += '[{},{}],'.format(_to_dg_time(dt_from_s(item.t, tz=item.tz)), data_part)

    if dg_data.endswith(','):
        dg_data = dg_data[0:-1]

    return global_min, global_max, '['+dg_data+']'


#=================
#  Dygraphs plot
#=================

def dygraphs_plot(timeseries, data_labels='all', data_indexes='all', aggregate=None, aggregate_by=None, full_precision=False,
                  color=None, height=None, image=DEFAULT_PLOT_AS_IMAGE, image_resolution='1280x380', html=False, save_to=None):
    """Plot a time series using Dygraphs interactive plots.
    
       Args:
           timeseries(DataTimePointSeries/DataTimeSlotSeries): the time series to plot.
           data_indexes(list): a list of data_labels to plot. By default set to all the data labels of the series.
           data_indexes(list): a list of data_indexes as the ``data_loss``, ``data_reconstructed`` etc.
                               to plot. By default set to all the data indexes of the series. To disable
                               plotting data indexes entirely, use None or an empty list.
           aggregate(bool): if to aggregate the time series, in order to speed up plotting.
                            By default, above 10000 data points the time series starts to
                            get aggregated by a factor of ten for each order of magnitude. 
           aggregate_by(int): a custom aggregation factor.
           full_precision(bool): if to use (nearly) full precision using 6 significant figures. Defaulted to false.
           color(str): the color of the time series in the plot. Only supported for univariate time series.
           height(int): force a plot height, in the time series data units.
           image(bool): if to generate an image rendering of the plot instead of the default interactive one.
                        To generate the image rendering, an headless Chromium web browser is downloaded on the
                        fly in order to render the plot as a PNG image.
           image_resolution(str): the image resolution, if generating an image rendering of the plot.
           html(bool): if to return the HTML code for the plot instead of generating an interactive or image one.
                       Useful for embedding it in a website or for generating multi-plot HTML pages.
           save_to(str): a file name (including its path) where to save the plot. If the plot is generated as 
                         interactive, then it is saved as a self-consistent HTML page which can be opened in a 
                         web browser. If the plot is generated as an image, then it is saved in PNG format.
    """
    # Credits: the interactive plot is based on the work here: https://www.stefaanlippens.net/jupyter-custom-d3-visualization.html.

    from IPython.display import display, Javascript, HTML, Image
    
    if html and image:
        raise ValueError('Setting both image and html to True is not supported.')

    if html and save_to:
        raise ValueError('Setting html=True is not compatible with setting a save_to value.')    
    
    if len(timeseries)==0:
        raise Exception('Cannot plot empty timeseries')
   
    if save_to:
        if image:
            if not save_to.endswith('.png'):
                logger.warning('You are saving to "{}" in image (PNG) format, but the file name does not end with ".png"'.format(save_to))
        else:
            if not save_to.endswith('.html'):
                logger.warning('You are saving to "{}" in interactive (HTML) format, but the file name does not end with ".html"'.format(save_to))
   
    if aggregate_by is None:
        # Set default aggregate_by if not explicitly set to something
        if len(timeseries)  > AGGREGATE_THRESHOLD:
            aggregate_by = 10**len(str(int(len(timeseries)/float(AGGREGATE_THRESHOLD))))
        else:
            aggregate_by = None
 
    if aggregate is not None:
        # Handle the case where we are required not to aggregate

        if not aggregate:
            aggregate_by = None

    #if show_data_reconstructed:
    #    show_data_loss=False

    # Handle series data_lables
    if data_labels == 'all':
        # Plot all the series data_labels
        data_labels_to_plot = timeseries.data_labels()
    elif data_labels is None:
        data_labels_to_plot = []
    else:
        # Check that the data_labels are of the right type and that are present in the series data_labels
        data_labels_to_plot = []
        if not isinstance(data_labels, list):
            raise TypeError('The "data_labels" argument must be a list or the magic word "all" (got type "{}")'.format(data_labels.__class__.__name__))
        for label in data_labels:
            if not isinstance(label, str):
                raise TypeError('The "data_labels" list items must be string (got type "{}")'.format(label.__class__.__name__))
            if not label in timeseries.data_labels():
                raise ValueError('The data label "{}" is not present in the series data labeles ({})'.format(label, timeseries._all_data_labels()))
            data_labels_to_plot.append(label)

    # Handle series data_indexes
    if data_indexes == 'all':
        # Plot all the series data_indexes
        data_indexes_to_plot = timeseries._all_data_indexes()
    elif data_indexes is None:
        data_indexes_to_plot = []
    else:
        # Check that the data_indexes are of the right type and that are present in the series data_indexes
        data_indexes_to_plot = []
        if not isinstance(data_indexes, list):
            raise TypeError('The "data_indexes" argument must be a list or the magic word "all" (got type "{}")'.format(data_indexes.__class__.__name__))
        for index in data_indexes:
            if not isinstance(index, str):
                raise TypeError('The "data_indexes" list items must be string (got type "{}")'.format(index.__class__.__name__))
            if not index in timeseries._all_data_indexes():
                raise ValueError('The data index "{}" is not present in the series data indexes ({})'.format(index, timeseries._all_data_indexes()))
            data_indexes_to_plot.append(index)

    # Checks
    if isinstance(timeseries, DataTimePointSeries):
        stepPlot_value   = 'false'
        drawPoints_value = 'true'
        if aggregate_by:
            legend_pre = 'Point (aggregated by {}) at '.format(aggregate_by)
        else:
            legend_pre = 'Point at '
    elif isinstance(timeseries, DataTimeSlotSeries):
        if isinstance(timeseries.resolution, TimeUnit):
            serie_unit_string = str(timeseries.resolution)
        else:
            serie_unit_string = str(timeseries.resolution) #.split('.')[0]+'s'
        #if aggregate_by: 
        #    raise NotImplementedError('Plotting slots with a plot-level aggregation is not yet supported')
        stepPlot_value   = 'true'
        drawPoints_value = 'false'
        if aggregate_by:
            # TODO: "Slot of {} unit" ?
            legend_pre = 'Slot of {}x{} (aggregated) starting at '.format(aggregate_by, serie_unit_string)
        else:
            legend_pre = 'Slot of {} starting at '.format(serie_unit_string)

    else:
        raise Exception('Don\'t know how to plot an object of type "{}"'.format(timeseries.__class__.__name__))

    # Set title
    if timeseries.title:
        title = timeseries.title
    else:
        if isinstance(timeseries, DataTimePointSeries):
            if aggregate_by:
                title = 'Time series of #{} points at {}, aggregated by {}'.format(len(timeseries), timeseries._resolution_string, aggregate_by)
            else:
                title = 'Time series of #{} points at {}'.format(len(timeseries), timeseries._resolution_string)
             
        elif isinstance(timeseries, DataTimeSlotSeries):
            if aggregate_by:
                # TODO: "slots of unit" ?
                title = 'Time series of #{} slots of {}, aggregated by {}'.format(len(timeseries), serie_unit_string, aggregate_by)
            else:
                # TODO: "slots of unit" ?
                title = 'Time series of #{} slots of {}'.format(len(timeseries), serie_unit_string)
                
        else:
            title = timeseries.__class__.__name__

    if aggregate_by:
        logger.info('Aggregating by "{}" for improved plotting'.format(aggregate_by))
            
    # Define div name
    graph_id = 'graph_' + str(uuid.uuid4()).replace('-', '')
    graph_div_id  = graph_id + '_plotarea'
    legend_div_id = graph_id + '_legend'

    # Do we have to show a series mark?
    if timeseries.mark:
        logger.info('Found series mark and showing it')
        if timeseries.mark_title:
            mark_title = timeseries.mark_title
        else:
            mark_title = 'mark'
        serie_mark_html = '<span style="background:rgba(255, 255, 102, .6);">&nbsp;{}&nbsp;</span>'.format(mark_title)
        serie_mark_html_off = '<span style="background:rgba(255, 255, 102, .2);">&nbsp;{}&nbsp;</span>'.format(mark_title)
    else:
        serie_mark_html=''
        serie_mark_html_off = ''

    # Dygraphs javascript
    dygraphs_javascript = """
// Taken and adapted from dygraph.js
function legendFormatter(data) {
      var g = data.dygraph;

      if (g.getOption('showLabelsOnHighlight') !== true) return '';
      
      var data_indexes = """+str(timeseries._all_data_indexes())+""";
      var sepLines = g.getOption('labelsSeparateLines');
      var first_data_index = true;
      var html;

      if (typeof data.x === 'undefined') {
        if (g.getOption('legend') != 'always') {
          return '';
        }
        // We are here when there's no selection and {legend: 'always'} is set.
        html = '"""+title+""": ';
        for (var i = 0; i < data.series.length; i++) {
          var series = data.series[i];
          // Skip not visible series
          if (!series.isVisible) continue;
          
          // Also skip Timeseria-specific series
          if (series.label=='data_mark') continue;
          
          // Add some initial stuff
          if (html !== '') html += sepLines ? '<br/>' : ' ';
          
          if (data_indexes.includes(series.label)){
              // The following was used to inlcude the "indexes" word in the legend
              /* if (first_data_index){
                  html += "<span style='margin-left:15px'>indexes:</span>"
                  first_data_index=false
              }*/
              html += "<span style='margin-left:15px; background: " + series.color + ";'>&nbsp;" + series.labelHTML + "&nbsp</span>, ";
          }
          else {
              html += "<span style='margin-left:15px; font-weight: bold; color: " + series.color + ";'>" + series.dashHTML + " " + series.labelHTML + "</span>, ";
          }
          
          // Remove last comma and space
          html = html.substring(0, html.length - 2);
        }
        html += ' &nbsp;&nbsp; """+serie_mark_html +"""'
        return html;
      }

      // Find out if we have an active mark
      mark_active = false
      for (var i = 0; i < data.series.length; i++) {
        if ( data.series[i].label=='data_mark' && data.series[i].y==1 )  {
           mark_active = true;
        }
      }


      html = '""" + legend_pre + """' + data.xHTML + ' (""" + str(timeseries.tz)+ """):';
      for (var i = 0; i < data.series.length; i++) {
        
        var mark_active = false;
        
        if ( data.series[i].label=='data_mark' && data.series[i].y==1 )  {
           mark_active = true;
        }
                 
        var series = data.series[i];
        
        // Skip not visible series
        if (!series.isVisible) continue;
        
        // Also skip Timeseria-specific series
        if (series.label=='data_mark') continue;
        
        if (sepLines) html += '<br>';
        //var decoration = series.isHighlighted ? ' class="highlight"' : '';
        var decoration = series.isHighlighted ? ' style="background-color: #000000"' : '';
        
        /*
        if (series.isHighlighted) {
            decoration = ' style="background-color: #fcf8b0"'
          }
        else{
            decoration = ' style="background-color: #ffffff"'
        }
        console.log(decoration)
        */
        
        //decoration = ' style="background-color: #fcf8b0"'
        if (data_indexes.includes(series.label)){
            // The following was used to inlcude the "indexes" word in the legend
            /*if (first_data_index){
                // Remove last comma and space
                html = html.substring(0, html.length - 2);
                html += "<span style='margin-left:15px'>indexes:</span>"
                first_data_index=false
            }*/
            html += "<span" + decoration + "> <span style='background: " + series.color + ";'>&nbsp" + series.labelHTML + "&nbsp</span>:&#160;" + """+ ('series.yHTML*100' if full_precision else '(Math.round(series.yHTML *100 * 100) / 100)')  +""" + "%</span>, ";
        
        }
        else {
            html += "<span" + decoration + "> <b><span style='color: " + series.color + ";'>" + series.labelHTML + "</span></b>:&#160;" + series.yHTML + "</span>, ";
        }
      
      }
      // Remove last comma and space
      html = html.substring(0, html.length - 2);
      
      if (mark_active) {
          html += ' &nbsp;"""+ serie_mark_html +"""'
      }
      //else {
      //    html += ' &nbsp;"""+ serie_mark_html_off +"""'
      //}
      return html;
    };
"""
    dygraphs_javascript += 'new Dygraph('
    dygraphs_javascript += 'document.getElementById("{}"),'.format(graph_div_id)

    # Check data for plot.
    _check_data_for_plot(timeseries[0].data)

    # Loop over all the keys
    labels=''
    if data_labels_to_plot:
        for data_label_to_plot in data_labels_to_plot:
            if isinstance(data_label_to_plot, int):
                if len(data_labels_to_plot) == 1:
                    labels='value,'
                else:    
                    labels += 'value {},'.format(data_label_to_plot+1)
            else:
                labels += '{},'.format(data_label_to_plot)
        # Remove last comma
        labels = labels[0:-1]
    else:
        labels='value'
    
    # Set index labels
    for index in data_indexes_to_plot:
        labels+=',{}'.format(index)

    # Handle series mark (as index)
    if timeseries.mark and RENDER_MARK_AS_INDEX:
        labels+=',data_mark'
    
    # Prepare labels string list for Dygraphs
    labels_list = ['Timestamp'] + labels.split(',')

    # Get series data in Dygraphs format (and aggregate if too much data and find global min and max)
    global_min, global_max, dg_data = _to_dg_data(timeseries, data_labels_to_plot, data_indexes_to_plot, full_precision, aggregate_by)
    if height:
        global_max = height
    if global_max != global_min:
        plot_min = str(global_min-((global_max-global_min)*0.1))
        plot_max = str(global_max+((global_max-global_min)*0.1))
    else:
        # Case where there is a straight line
        plot_min = str(global_min*0.99)
        plot_max = str(global_max*1.01)

    #dygraphs_javascript += '"Timestamp,{}\\n{}",'.format(labels, dg_data)
    dygraphs_javascript += '{},'.format(dg_data)

#dateWindow: [0,"""+str(_utc_fake_s_from_dt(timeseries[-1].end.dt)*1000)+"""],

    dygraphs_javascript += """{\
labels: """+str(labels_list)+""",
drawGrid: true,
drawPoints:"""+drawPoints_value+""",
strokeWidth: 1.5,
pointSize:1.5,
highlightCircleSize:4,
stepPlot: """+stepPlot_value+""",
fillGraph: false,
fillAlpha: 0.5,
colorValue: 0.5,
showRangeSelector: """+('true' if not image else 'false')+""",
//rangeSelectorHeight: 30,
hideOverlayOnMouseOut: true,
interactionModel: Dygraph.defaultInteractionModel,
includeZero: false,
digitsAfterDecimal:2,
sigFigs: """ + ('6' if full_precision else 'null') + """,  
showRoller: false,
legend: 'always',
labelsUTC: true,
legendFormatter: legendFormatter,
labelsDiv: '"""+legend_div_id+"""',
valueRange: ["""+plot_min+""", """+plot_max+"""],
zoomCallback: function() {
    this.updateOptions({zoomRange: ["""+plot_min+""", """+plot_max+"""]});
},
axes: {
  /*y: {
    // set axis-related properties here
    drawGrid: false,
    independentTicks: true
  },*/
  y2: {
    drawGrid: false,
    drawAxis: false,
    axisLabelWidth:0,
    independentTicks: true,
    valueRange: [0,1.01],
    //customBars: false, // Does not work?
  }/*,
  x : {
                    valueFormatter: Dygraph.dateString_,
                    valueParser: function(x) { return new Date(x); },
 }*/
},
animatedZooms: true,"""

    # Variable fill alpha in case of aggregated plot or not (for the area)
    # They have to be differentiated due different transparency policies in Dygraphs
    if aggregate_by:        
        fill_alpha_value  = 0.31
    else:
        fill_alpha_value  = 0.5

    # Fixed fill alpha value, used for some colors
    fill_alpha_value_fixed = 0.6

    # Fixed colors (alpha is for the lgend)
    rgba_value_red    = 'rgba(255,128,128,0.4)'
    rgba_alpha_violet = 'rgba(227, 168, 237, 0.5)' 
    rgba_value_yellow = 'rgba(255, 255, 102, 0.6)'
    rgba_value_darkorange = 'rgba(255, 140, 0, 0.7)'
    
    # Data indexes series start
    dygraphs_javascript += """
     series: {"""
    
    colormap_count = 0
    for i, data_index_to_plot in enumerate(data_indexes_to_plot):
    
        # Special data indexes
        if data_index_to_plot == 'data_reconstructed':
            data_index_fillalpha = fill_alpha_value_fixed
            data_index_color = rgba_alpha_violet

        elif data_index_to_plot == 'data_loss':
            data_index_fillalpha = fill_alpha_value
            data_index_color = rgba_value_red

        elif data_index_to_plot == 'forecast':
            data_index_fillalpha = fill_alpha_value
            data_index_color = rgba_value_yellow

        elif data_index_to_plot ==  'anomaly':
            data_index_fillalpha = fill_alpha_value_fixed
            data_index_color = rgba_value_darkorange
        
        # Standard data indexes
        else:
            data_index_fillalpha = fill_alpha_value-0.1
            data_index_color = to_rgba_str_from_norm_rgb(_tab10_colormap_norm[colormap_count%len(_tab10_colormap_norm)], fill_alpha_value+0.1 if aggregate_by else fill_alpha_value-0.1)
            colormap_count += 1
           
        # Now create the entry for this data index               
        dygraphs_javascript += """
        '"""+data_index_to_plot+"""': {
         //customBars: false, // Does not work?
         axis: 'y2',
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: """+str(data_index_fillalpha)+""", // This alpha is used for the area
         color: '"""+data_index_color+"""'             // Alpha here is used for the legend 
       },"""    


    # Add data mark index series
    dygraphs_javascript += """
       'data_mark': {
         //customBars: false, // Does not work?
         axis: 'y2',
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: """+str(fill_alpha_value)+""",  // This alpha is used for the area
         color: '"""+rgba_value_yellow+"""'         // Alpha here is used for the legend 
       },"""

    # Add all non-index series to be included in the miniplot
    for label in data_labels_to_plot:
        dygraphs_javascript += """
           '"""+str(label)+"""': {
             showInRangeSelector:true
           },"""     
    
    # Special series end
    dygraphs_javascript += """
     },"""

    # Force "original" Dygraph color if only one data series, or use custom color:          
    if len(data_labels_to_plot) <=1:
        if color:
            dygraphs_javascript += """colors: ['"""+color+"""'],"""         
        else:
            dygraphs_javascript += """colors: ['rgb(0,128,128)'],""" 

    # Handle series mark (only if not handled as an index)
    if timeseries.mark and not RENDER_MARK_AS_INDEX:
 
        # Check that we know how to use this mark
        if not isinstance(timeseries.mark[0], datetime.datetime):
            raise TypeError('Series marks must be datetime objects')
        if not isinstance(timeseries.mark[1], datetime.datetime):
            raise TypeError('Series marks must be datetime objects')            
           
        # Convert the mark to fake epoch milliseconds
        mark_start = _utc_fake_s_from_dt(timeseries.mark[0])*1000
        mark_end   = _utc_fake_s_from_dt(timeseries.mark[1])*1000
         
        # Add js code
        dygraphs_javascript += 'underlayCallback: function(canvas, area, g) {'
        dygraphs_javascript += 'var bottom_left = g.toDomCoords({}, -20);'.format(mark_start)
        dygraphs_javascript += 'var top_right = g.toDomCoords({}, +20);'.format(mark_end)
        dygraphs_javascript += 'var left = bottom_left[0];'
        dygraphs_javascript += 'var right = top_right[0];'
        dygraphs_javascript += 'canvas.fillStyle = "rgba(255, 255, 102, .5)";'
        dygraphs_javascript += 'canvas.fillRect(left, area.y, right - left, area.h);'
        dygraphs_javascript += '}'

    if aggregate_by:
        # Add a cooma. Just if the previosu section kiked-in
        if dygraphs_javascript[-1] != ',':
            dygraphs_javascript += ','
        dygraphs_javascript += 'customBars: true'        
    dygraphs_javascript += '})'
    
    #if log_js:
    #    logger.info(dygraphs_javascript)

    rendering_javascript = """
require.undef('"""+graph_id+"""'); // Helps to reload in Jupyter
define('"""+graph_id+"""', ['dgenv'], function (Dygraph) {

    function draw(div_id) {
    """ + dygraphs_javascript +"""
    }
    return draw;
});
"""

    STATIC_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/static/'

    # By defualt we show the plot
    show_plot = True

    if image or save_to or html:

        # Generate random UUID
        rnd_uuid=str(uuid.uuid4())
        
        # If a destination was set, we do not show the plot.
        if save_to:
            show_plot = False
        
        # Set destination file values
        if save_to:
            if image:
                # Dump as image
                html_dest = '/tmp/{}.html'.format(rnd_uuid)
                png_dest = save_to
            else:
                # Interactive, dump as html
                html_dest = save_to
                png_dest=None
        else:
            if image:
                # Dump as image
                html_dest = '/tmp/{}.html'.format(rnd_uuid)
                png_dest = '/tmp/{}.png'.format(rnd_uuid)
            else:
                # Will never get here, directly rendered in iPython
                pass

        # Start building HTML content
        html_content = ''
        if not html:
            html_content += '<html><head>'
            with open(STATIC_DATA_PATH+'/js/dygraph-2.1.0.min.js') as dg_js_file:
                html_content += '<script type="text/javascript">'+dg_js_file.read()+'</script>\n'
            with open(STATIC_DATA_PATH+'css/dygraph-2.1.0.css') as dg_css_file:
                html_content += '<style>'+dg_css_file.read()+'</style>\n'
            html_content += '</head><body style="font-family:\'Helvetica Neue\', Helvetica, Arial, sans-serif; font-size:1.0em">\n'
            html_content += '<div style="height:36px; padding:0; margin-left:0px; margin-top:10px">\n'
        html_content += '<div id="{}" style="width:100%"></div></div>\n'.format(legend_div_id)
        html_content += '<div id="{}" style="width:100%; margin-right:0px"></div>\n'.format(graph_div_id)
        if not html:
            html_content += '</body></html>\n'
        html_content += '<script>{}</script>\n'.format(dygraphs_javascript)
        
        # Dump html to file
        try:
            with open(html_dest, 'w') as f:
                f.write(html_content)
        except UnboundLocalError:
            pass
        
        # Also render if not interactive mode
        if image:

            if not image_plot_support:
                raise ValueError('Sorry, image plots are not supported since the package "pyppeteer" could be imported')

            # Get OS and architecture
            _os = os.uname()[0]
            _arch = os.uname()[4]
            
            # For ARM and Linux, use system system chromium, otherwise rely on Pyppeteer

            if (_os == 'Linux') and (_arch == 'aarch64'):
                _chromium_executable = 'chromium-browser'
            else:
                # Disable logging for Pyppeteer
                pyppeteer_logger = logging.getLogger('pyppeteer')
                pyppeteer_logger.setLevel(logging.CRITICAL)
                
                # Check we have Chromium,. if not, download
                if not chromium_executable().exists():
                    logger.info('Downloading Chromium for rendering image-based plots...')
                    download_chromium()
                    logger.info('Done. Will not be required anymore on this system.')
                _chromium_executable = chromium_executable()

            # Render HTML to image
            resolution = image_resolution.replace('x', ',')
            command = '{} --no-sandbox --headless --disable-gpu --window-size={} --screenshot={} {}'.format(_chromium_executable, resolution, png_dest, html_dest)
            logger.debug('Executing "{}"'.format(command))
            out = os_shell(command, capture=True)
            
            if not out.exit_code == 0:
                raise OSError('Error: {}'.format(out.stderr))
            
            if show_plot:
                return (Image(filename=png_dest))
        
        if html:
            return html_content
        
        # Log 
        if save_to:
            if image:
                logger.info('Saved image plot in PNG format to "{}"'.format(png_dest))
            else:
                logger.info('Saved interactive plot in HTML format to "{}"'.format(html_dest))
                    
    else:

        if show_plot:
            # Require Dygraphs Javascript library
            # https://raw.githubusercontent.com/sarusso/Timeseria/develop/timeseria/static/js/dygraph-2.1.0.min
            display(Javascript("require.config({paths: {dgenv: 'https://cdnjs.cloudflare.com/ajax/libs/dygraph/2.1.0/dygraph.min'}});"))
            # TODO: load it locally, maybe with something like:    
            #with open(STATIC_DATA_PATH+'js/dygraph-2.1.0.min.js') as dy_js_file:
            #    display(Javascript(dy_js_file.read()))
        
            # Load Dygraphs CSS
            with open(STATIC_DATA_PATH+'css/dygraph-2.1.0.css') as dg_css_file:
                display(HTML('<style>'+dg_css_file.read()+'</style>'))
            
            # Load Dygraphs Javascript and html code    
            display(Javascript(rendering_javascript))
            display(HTML('''<div style="height:36px; padding:0; margin-left:0px; margin-top:10px">
                            <div id="'''+legend_div_id+'''" style="width:100%"></div></div>
                            <div id="'''+graph_div_id+'''" style="width:100%; margin-right:0px"></div>'''))
        
            # Render the graph (this is the "entrypoint")
            display(Javascript("""
                (function(element){
                    require(['"""+graph_id+"""'], function("""+graph_id+""") {
                        """+graph_id+"""(element.get(0), '%s');
                    });
                })(element);
            """ % (graph_div_id)))


#=================
# Matplotlib plot
#=================

def matplotlib_plot(timeseries):
    """Plot a timeseries in Jupyter using Matplotlib."""
    from matplotlib import pyplot
    pyplot.plot(timeseries.df)
    pyplot.show()
