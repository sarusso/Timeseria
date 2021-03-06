import os
import uuid
import datetime
from .time import dt_from_s, dt_to_str, utckfake_s_from_dt
from .datastructures import DataTimePointSeries, DataTimeSlotSeries
from .units import TimeUnit
from .utilities import is_numerical

# Setup logging
import logging
logger = logging.getLogger(__name__)

''' Note: the code in this module is spaghetti-code, and there are no tests. A major refactoring is required.'''

INDEX_METRICS = ['anomaly']

#=================
#   Utilities
#=================

def get_keys_and_check_data_for_plot(data):
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
def to_dg_time(dt):
    '''Get Dygraphs time form datetime'''
    #return '{}{:02d}-{:02d}T{:02d}:{:02d}:{:02d}+00:00'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute,dt.second)
    #return s_from_dt(dt)*1000
    if dt.microsecond:
        return 'new Date(Date.UTC({}, {}, {}, {}, {}, {}, {}))'.format(dt.year, dt.month-1, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)
    else:
        return 'new Date(Date.UTC({}, {}, {}, {}, {}, {}))'.format(dt.year, dt.month-1, dt.day, dt.hour, dt.minute, dt.second)


def to_dg_data(serie, aggregate_by=0, plot_data_loss=False, plot_data_reconstructed=False, serie_mark=None):
    '''{% for timestamp,value in metric_data.items %}{{timestamp}},{{value}}\n{%endfor%}}'''
    
    dg_data=''
    first_t  = None
    last_t   = None

    global_min = None
    global_max = None
    
    if serie_mark:
        serie_mark_start_t = s_from_dt(serie.mark[0])
        serie_mark_end_t = s_from_dt(serie.mark[1])

    for i, item in enumerate(serie):
        
        # Offset
        i=i+1
                
        # Prepare a dict for every piece of data
        try:
            keys
        except UnboundLocalError:
            # Set keys and support vars
            keys = get_keys_and_check_data_for_plot(item.data)
            if not keys:
                keys = [None]
            data_sums = [0 for key in keys]
            data_mins = [None for key in keys]
            data_maxs = [None for key in keys]
            data_loss = None
            data_reconstructed = None

        if aggregate_by:

            if first_t is None:
                if isinstance(serie, DataTimePointSeries):
                    first_t = item.t
                elif isinstance(serie, DataTimeSlotSeries):
                    first_t = item.start.t
            
            # Prepare to exclude index metrics from the min/max computation
            index_metrics_positions = []
                  
            # Define data
            if keys:
                datas=[]      
                for j, key in enumerate(keys):
                    if key is None:
                        datas.append(item.data)
                    else:
                        if key in INDEX_METRICS:
                            index_metrics_positions.append(j) 
                        datas.append(item.data[key])

            else:
                datas=[item.data]
            
            for j, data in enumerate(datas):
                
                # Sum
                data_sums[j] += data
                
                if j not in index_metrics_positions:
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
            
            #if isinstance(serie, DataTimeSlotSeries):
            # Data loss
            if item.data_loss is not None:
                if data_loss is None:
                    data_loss = 0
                data_loss += item.data_loss 
            
            # Data reconstructed
            if plot_data_reconstructed:
                if item._data_reconstructed is not None:
                    if data_reconstructed is None:
                        data_reconstructed = 0
                    data_reconstructed += item._data_reconstructed
    
            # Dump aggregated data?
            if (i!=0)  and ((i % aggregate_by)==0):
                if isinstance(serie, DataTimePointSeries):
                    last_t = item.t
                elif isinstance(serie, DataTimeSlotSeries):
                    last_t = item.end.t
                t = first_t + ((last_t-first_t) /2)
                data_part=''
                for i, key in enumerate(keys):
                    avg = data_sums[i]/aggregate_by
                    if i  in index_metrics_positions:
                        data_part+='[{},{},{}],'.format( 0, avg, avg)
                    else:
                        data_part+='[{},{},{}],'.format( data_mins[i], avg, data_maxs[i])

                # Do we have a mark?
                if serie_mark:
                    if item.start.t >= serie_mark_start_t and item.end.t < serie_mark_end_t:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            data_part+='[0,{0},{0}],'.format(data_reconstructed/aggregate_by)                   
                        # Plot data loss
                        if plot_data_loss:
                            if data_loss is not None:
                                data_part+='[0,{0},{0}],'.format(data_loss/aggregate_by)
                            else:
                                data_part+='[,,],'
                                    
                        # Add the (active) mark
                        data_part+='[0,1,1],'
                        
                    else:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            data_part+='[0,{0},{0}],'.format(data_reconstructed/aggregate_by)  
                        # Plot data loss
                        if plot_data_loss:
                            if data_loss is not None:
                                data_part+='[0,{0},{0}],'.format(data_loss/aggregate_by)
                            else:
                                data_part+='[,,],'
                            
                        # Add the (inactive) mark
                        data_part+='[0,0,0],'
                            
                else:
                    # Plot data reconstructed
                    if plot_data_reconstructed:
                        data_part+='[0,{0},{0}],'.format(data_reconstructed/aggregate_by)  
                    # Plot data loss
                    if plot_data_loss:
                        if data_loss is not None:
                            data_part+='[0,{0},{0}],'.format(data_loss/aggregate_by)
                        else:
                            data_part+=','

                # Remove last comma
                data_part=data_part[0:-1]
                dg_data+='[{},{}],'.format(to_dg_time(dt_from_s(t, tz=item.tz)), data_part)
                
                # Reset averages
                data_sums = [0 for key in keys]
                data_mins = [None for key in keys]
                data_maxs = [None for key in keys]
                data_loss = None
                data_reconstructed = None
                first_t = None

        #====================
        #  No aggregation
        #====================
        else:
                    
            # Loop over all the keys, including the "None" key if we have no keys (float data)
            data_part=''
            for key in keys:
                if key is None:
                    data = item.data
                else:
                    data = item.data[key]

                data_part += '{},'.format(data)   #item.data
                
                # Exclude some special stuff from the min/max computation

                if key not in INDEX_METRICS:

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
                

            if isinstance(serie, DataTimePointSeries):
                t = item.t

                # Do we have a mark?
                if serie_mark:
 
                    if item.t >= serie_mark_start_t and item.t < serie_mark_end_t:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            if item._data_reconstructed is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item._data_reconstructed)
                        # Plot data data loss
                        if plot_data_loss:
                            if item.data_loss is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item.data_loss)
                        data_part+='1,'
                        
                    else:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            if item._data_reconstructed is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item._data_reconstructed)
                        # Plot data loss
                        if plot_data_loss:
                            if item.data_loss is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item.data_loss)                        
                        data_part+='0,'
                else:
                    # Plot data reconstructed
                    if plot_data_reconstructed:
                        if item._data_reconstructed is None:
                            data_part+=','
                        else:
                            data_part+='{},'.format(item._data_reconstructed)
                    # Plot data loss
                    if plot_data_loss:
                        if item.data_loss is None:
                            data_part+=','
                        else:
                            data_part+='{},'.format(item.data_loss)

                        
                    
            
            
            elif isinstance(serie, DataTimeSlotSeries):
                t = item.start.t

                # Do we have a mark?
                if serie_mark:
 
                    if item.start.t >= serie_mark_start_t and item.end.t < serie_mark_end_t:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            if item._data_reconstructed is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item._data_reconstructed)
                        # Plot data loss
                        if plot_data_loss:
                            if item.data_loss is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item.data_loss)
                        data_part+='1,'
                        
                    else:
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            if item._data_reconstructed is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item._data_reconstructed)
                        # Plot data loss
                        if plot_data_loss:
                            if item.data_loss is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item.data_loss)

                                                
                        data_part+='0,'
                else:
                    if isinstance(serie, DataTimeSlotSeries):
                        # Plot data reconstructed
                        if plot_data_reconstructed:
                            if item._data_reconstructed is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item._data_reconstructed)
                        # Plot data loss
                        if plot_data_loss:
                            if item.data_loss is None:
                                data_part+=','
                            else:
                                data_part+='{},'.format(item.data_loss)                          
                
            # Remove last comma
            data_part = data_part[0:-1]



            dg_data+='[{},{}],'.format(to_dg_time(dt_from_s(t, tz=item.tz)), data_part)

    if dg_data.endswith(','):
        dg_data = dg_data[0:-1]

    return global_min, global_max, '['+dg_data+']'


#=================
#  Dygraphs plot
#=================

def dygraphs_plot(serie, aggregate_by, log_js=False, show_data_loss=True, show_forecasted=True, show_data_reconstructed=False):
    '''Plot a data_time_pointSeries in Jupyter using Dugraph. Based on the work here: https://www.stefaanlippens.net/jupyter-custom-d3-visualization.html'''
    from IPython.display import display, Javascript, HTML
    
    if len(serie)==0:
        raise Exception('Cannot plot empty serie')

    #if show_data_reconstructed:
    #    show_data_loss=False

    # Checks
    if isinstance(serie, DataTimePointSeries):
        stepPlot_value   = 'false'
        drawPoints_value = 'true'
        if aggregate_by:
            legend_pre = 'Point (aggregated by {}) at '.format(aggregate_by)
        else:
            legend_pre = 'Point at '
    elif isinstance(serie, DataTimeSlotSeries):
        if isinstance(serie._resolution, TimeUnit):
            serie_unit_string = str(serie._resolution)
        else:
            serie_unit_string = str(serie._resolution) #.split('.')[0]+'s'
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
        raise Exception('Don\'t know how to plot an object of type "{}"'.format(serie.__class__.__name__))

    # Set title
    if serie.title:
        title = serie.title
    else:
        if isinstance(serie, DataTimePointSeries):
            if aggregate_by:
                title = 'Time series of #{} points, aggregated by {}'.format(len(serie), aggregate_by)
            else:
                title = 'Time series of #{} points'.format(len(serie))
             
        elif isinstance(serie, DataTimeSlotSeries):
            if aggregate_by:
                # TODO: "slots of unit" ?
                title = 'Time series of #{} slots of {}, aggregated by {}'.format(len(serie), serie_unit_string, aggregate_by)
            else:
                # TODO: "slots of unit" ?
                title = 'Time series of #{} slots of {}'.format(len(serie), serie_unit_string)
                
        else:
            title = serie.__class__.__name__

    if aggregate_by:
        logger.info('Aggregating by "{}" for improved plotting'.format(aggregate_by))
            
    # Define div name
    graph_id = 'graph_' + str(uuid.uuid4()).replace('-', '')
    graph_div_id  = graph_id + '_plotarea'
    legend_div_id = graph_id + '_legend'

    # Mark to use?
    try:
        if show_forecasted and serie.mark:
            if not isinstance(serie.mark, (list, tuple)):
                raise TypeError('Series mark must be a list or tuple')
            if not len(serie.mark) == 2:
                raise ValueError('Series mark must be a list or tuple of two elements')
            if not isinstance(serie.mark[0], datetime.datetime):
                raise TypeError('Series marks must be datetime objects')
            if not isinstance(serie.mark[1], datetime.datetime):
                raise TypeError('Series marks must be datetime objects')
            logger.info('Found data forecast mark and showing it')
            serie_mark=True
            serie_mark_html = '<span style="background:rgba(255, 255, 102, .6);">&nbsp;forecast&nbsp;</span>'
            serie_mark_html_off = '<span style="background:rgba(255, 255, 102, .2);">&nbsp;forecast&nbsp;</span>'
        else:
            serie_mark=False
            serie_mark_html=''
            serie_mark_html_off = ''
    except AttributeError:
        serie_mark=False
        serie_mark_html = ''
        serie_mark_html_off=''

    # Dygraphs javascript
    dygraphs_javascript = """
// Taken and adapted from dygraph.js
function legendFormatter(data) {
      var g = data.dygraph;

      if (g.getOption('showLabelsOnHighlight') !== true) return '';

      var sepLines = g.getOption('labelsSeparateLines');
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
          

          if ((series.label=='data_reconstructed') || (series.label=='data_loss') || (series.label=='anomaly')){
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

      // Find out if we have a forcast
      mark_as_forecast = false
      for (var i = 0; i < data.series.length; i++) {
        if ( data.series[i].label=='data_mark' && data.series[i].y==1 )  {
           mark_as_forecast = true;
        }
      }


      html = '""" + legend_pre + """' + data.xHTML + ' (""" + str(serie.tz)+ """):';
      for (var i = 0; i < data.series.length; i++) {
        
        var mark_as_forecast = false;
        
        if ( data.series[i].label=='data_mark' && data.series[i].y==1 )  {
           mark_as_forecast = true;
           console.log('forecast')
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
        if ((series.label=='data_reconstructed') || (series.label=='data_loss') || (series.label=='anomaly')){
            html += "<span" + decoration + "> <span style='background: " + series.color + ";'>&nbsp" + series.labelHTML + "&nbsp</span>:&#160;" + series.yHTML*100 + "%</span>, ";
        
        }
        else {
            html += "<span" + decoration + "> <b><span style='color: " + series.color + ";'>" + series.labelHTML + "</span></b>:&#160;" + series.yHTML + "</span>, ";
        }
      
      }
      // Remove last comma and space
      html = html.substring(0, html.length - 2);
      
      if (mark_as_forecast) {
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

    # Loop over all the keys
    labels=''
    keys = get_keys_and_check_data_for_plot(serie[0].data)
    if keys:
        for key in keys:
            if isinstance(key, int):
                if len(keys) == 1:
                    labels='value,'
                else:    
                    labels += 'value {},'.format(key+1)
            else:
                labels += '{},'.format(key)
        # Remove last comma
        labels = labels[0:-1]
    else:
        labels='value'

    data_reconstructed_indexes = False
    data_loss_indexes          = False
    #if isinstance(serie, DataTimeSlotSeries):
    # Do we have data reconstructed or losses indexes?
    try:
        if serie[0]._data_reconstructed is not None:
            data_reconstructed_indexes = True
    except:
        data_reconstructed_indexes = False
    if serie[0].data_loss is not None:
        data_loss_indexes = True

    plot_data_reconstructed = False
    plot_data_loss = False
    if show_data_reconstructed and data_reconstructed_indexes:
        logger.info('Found data reconstruction index and showing it')
        labels+=',data_reconstructed'
        plot_data_reconstructed = True
    if show_data_loss and data_loss_indexes:
        labels+=',data_loss'
        plot_data_loss = True
    if serie_mark:
        labels+=',data_mark'
    
        
    labels_list = ['Timestamp'] + labels.split(',')

    global_min, global_max, dg_data = to_dg_data(serie, aggregate_by, plot_data_loss=plot_data_loss, plot_data_reconstructed=plot_data_reconstructed, serie_mark=serie_mark)
    if global_max != global_min:
        plot_min = str(global_min-((global_max-global_min)*0.1))
        plot_max = str(global_max+((global_max-global_min)*0.1))
    else:
        # Case where there is a straight line
        plot_min = str(global_min*0.99)
        plot_max = str(global_max*1.01)

    #dygraphs_javascript += '"Timestamp,{}\\n{}",'.format(labels, dg_data)
    dygraphs_javascript += '{},'.format(dg_data)

#dateWindow: [0,"""+str(utckfake_s_from_dt(serie[-1].end.dt)*1000)+"""],

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
showRangeSelector: true,
//rangeSelectorHeight: 30,
hideOverlayOnMouseOut: true,
interactionModel: Dygraph.defaultInteractionModel,
includeZero: false,
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

    if aggregate_by:
        # Alpha is for the legend
        rgba_value_red    = 'rgba(255,128,128,0.4)'
        rgba_value_gray   = 'rgba(240,240,240,0.5)'
        rgba_value_orange = 'rgba(235, 156, 56,1.0)'
        fill_alpha_value  = 0.31 # 31 Alpha for the area (will not be used for rgba_value_orange)
        fill_alpha_value_orange = 0.6
    else:
        # Alpha is for the legend
        rgba_value_red    = 'rgba(255,128,128,0.4)' 
        rgba_value_gray   = 'rgba(240,240,240,0.5)'
        rgba_value_orange = 'rgba(245, 193, 130,0.8)'
        fill_alpha_value  = 0.5  # Alpha for the area (will not be used for rgba_value_orange)
        fill_alpha_value_orange = 0.6
        
    rgba_value_yellow = 'rgba(255, 255, 102, 1)'
    rgba_value_darkorange = 'rgba(255, 140, 0, 0.7)'
    
    
    # Special series
    dygraphs_javascript += """
     series: {"""
    
    if show_data_reconstructed and data_reconstructed_indexes:
        # Add data reconstructed special serie
        dygraphs_javascript += """
       'data_reconstructed': {
         //customBars: false, // Does not work?
         axis: 'y2',
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: """+str(fill_alpha_value_orange)+""",            // This alpha is used for the area
         color: '"""+rgba_value_orange+"""'
       },"""
       
    if show_data_loss and data_loss_indexes:
        # Add data loss special serie
        dygraphs_javascript += """
       'data_loss': {
         //customBars: false, // Does not work?
         axis: 'y2',
         //stepPlot: true,
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: """+str(fill_alpha_value)+""",  // This alpha is used for the area 
         color: '"""+rgba_value_red+"""'            // Alpha here is used for the legend 
       },"""
    
    # Add all series to be included in the miniplot,
    for label in serie.data_keys():
        dygraphs_javascript += """
           '"""+str(label)+"""': {
             showInRangeSelector:true
           },"""     
    
    # Add data mark and anomalay series
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
       },
       'anomaly': {
         //customBars: false, // Does not work?
         axis: 'y2',
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: 0.6,  // This alpha is used for the area
         color: '"""+rgba_value_darkorange+"""'     // Alpha here is used for the legend 
       },
     },
    """
    
    # Force "original" Dygraph color.
    labels_excluding_index_metrics= 0
    if True: #isinstance(serie, DataTimeSlotSeries):    
        for key in serie.data_keys():
            if key not in INDEX_METRICS:
                labels_excluding_index_metrics +=1
                    
        if labels_excluding_index_metrics <=1:
            dygraphs_javascript += """colors: ['rgb(0,128,128)'],""" 

    # Plotting series mark is disabled for now as we use the data_mark series instead
    if serie_mark and False:            
        # Convert the mark to fake epoch milliseconds
        mark_start = utckfake_s_from_dt(serie.mark[0])*1000
        mark_end   = utckfake_s_from_dt(serie.mark[1])*1000
        # Add js code
        dygraphs_javascript  += 'underlayCallback: function(canvas, area, g) {'
        dygraphs_javascript += 'var bottom_left = g.toDomCoords({}, -20);'.format(mark_start)
        dygraphs_javascript += 'var top_right = g.toDomCoords({}, +20);'.format(mark_end)
        dygraphs_javascript += 'var left = bottom_left[0];'
        dygraphs_javascript += 'var right = top_right[0];'
        dygraphs_javascript += 'canvas.fillStyle = "rgba(255, 255, 102, .5)";'
        dygraphs_javascript += 'canvas.fillRect(left, area.y, right - left, area.h);'
        dygraphs_javascript += '}'

    if aggregate_by:
        dygraphs_javascript += 'customBars: true'        
    dygraphs_javascript += '})'
    
    if log_js:
        logger.info(dygraphs_javascript)

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
    display(HTML('<div style="height:20px; margin-left:0px"><div id="{}" style="width:100%"></div></div>'.format(legend_div_id)))
    display(HTML('<div id="{}" style="width:100%; margin-right:0px"></div>'.format(graph_div_id)))

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

def matplotlib_plot(serie):
    def as_vectors(self):
        X = [datetime.datetime.fromtimestamp(item.t) for item in serie]
        Y = [item.data for item in serie]
        return X,Y
    from matplotlib import pyplot
    pyplot.plot(*as_vectors(serie))
    pyplot.show()
