import os
import uuid
import datetime
from .time import dt_from_s
from .datastructures import DataTimePointSerie, DataTimeSlotSerie

# Setup logging
import logging
logger = logging.getLogger(__name__)

''' Note: the code in this module is spaghetti-code, and there are no tests. A major refactoring is required.'''

#=================
#   Utilities
#=================

def is_numerical(item):
    if isinstance(item, float):
        return True
    if isinstance(item, int):
        return True
    return False


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


def to_dg_time(dt):
    '''Get Dygraphs time form datetime'''
    return '{}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute,dt.second)


def to_dg_data(serie, aggregate_by=0):
    '''{% for timestamp,value in metric_data.items %}{{timestamp}},{{value}}\n{%endfor%}}'''
    
    dg_data=''
    first_t  = None
    last_t   = None

    global_min = None
    global_max = None

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

        if aggregate_by:

            if first_t is None:
                if isinstance(serie, DataTimePointSerie):
                    first_t = item.t
                elif isinstance(serie, DataTimeSlotSerie):
                    first_t = item.start.t
                
                  
            # Define data
            data_loss_index = None
            if keys:
                datas=[]      
                for j, key in enumerate(keys):
                    if data_loss_index is None:
                        if key == 'data_loss':
                            data_loss_index = j
                    if key is None:
                        datas.append(item.data)
                    else:
                        datas.append(item.data[key])

            else:
                datas=[item.data]
            
            for j, data in enumerate(datas):
                
                # Sum
                data_sums[j] += data
                
                # Global min
                if j != data_loss_index:
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

            # Dump aggregated data?
            if (i!=0)  and ((i % aggregate_by)==0):
                if isinstance(serie, DataTimePointSerie):
                    last_t = item.t
                elif isinstance(serie, DataTimeSlotSerie):
                    last_t = item.end.t
                t = first_t + ((last_t-first_t) /2)
                data_part=''
                for i, key in enumerate(keys):
                    avg = data_sums[i]/aggregate_by
                    if key == 'data_loss':
                        data_part+='{};{};{},'.format(0, avg, avg)                    
                    else:
                        data_part+='{};{};{},'.format( data_mins[i], avg, data_maxs[i])
                data_part=data_part[0:-1]
                dg_data+='{},{}\\n'.format(to_dg_time(dt_from_s(t)), data_part)
                data_sums = [0 for key in keys]
                data_mins = [None for key in keys]
                data_maxs = [None for key in keys]
                first_t = None

        else:
                    
            # Loop over all the keys, including the "None" key if we have no keys (float data)
            data_part=''
            for key in keys:
                if key is None:
                    data = item.data
                else:
                    data = item.data[key]

                data_part += '{},'.format(data)   #item.data

                if key != 'data_loss':         
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
                
            # Remove last comma
            data_part = data_part[0:-1]

            if isinstance(serie, DataTimePointSerie):
                t = item.t
            elif isinstance(serie, DataTimeSlotSerie):
                t = item.start.t

            dg_data+='{},{}\\n'.format(to_dg_time(dt_from_s(t)), data_part)

    return global_min, global_max, dg_data


#=================
#  Dygraphs plot
#=================

def dygraphs_plot(serie, aggregate_by):
    '''Plot a dataTimePointSeries in Jupyter using Dugraph. Based on the work here: https://www.stefaanlippens.net/jupyter-custom-d3-visualization.html'''
    from IPython.display import display, Javascript, HTML

    if len(serie)==0:
        raise Exception('Cannot plot empty serie')

    # Checks
    if isinstance(serie, DataTimePointSerie):
        stepPlot_value   = 'false'
        drawPoints_value = 'true'
        legend_pre = ''
    elif isinstance(serie, DataTimeSlotSerie):
        stepPlot_value   = 'true'
        drawPoints_value = 'false'
        if aggregate_by:
            legend_pre = 'Slot of {}x{} starting at '.format(aggregate_by, serie.timeSpan)
        else:
            legend_pre = 'Slot of {} starting at '.format(serie.timeSpan)

    else:
        raise Exception('Don\'t know how to plot an object of type "{}"'.format(serie.__class__.__name__))

    # Set title
    if serie.title:
        title = serie.title
    else:
        if isinstance(serie, DataTimePointSerie):
            title = serie.__class__.__name__
        elif isinstance(serie, DataTimeSlotSerie):
            title = '{} of {} Slots '.format(serie.__class__.__name__, serie.timeSpan)
        else:
            title = serie.__class__.__name__

    if aggregate_by:
        logger.info('Aggregating by "{}" for plotting'.format(aggregate_by))
            
    # Define div name
    graph_id = 'graph_' + str(uuid.uuid4()).replace('-', '')
    graph_div_id  = graph_id + '_plotarea'
    legend_div_id = graph_id + '_legend'


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

          if (html !== '') html += sepLines ? '<br/>' : ' ';
          html += "<span style='margin-left:15px; font-weight: bold; color: " + series.color + ";'>" + series.dashHTML + " " + series.labelHTML + "</span>, ";
          // Remove last comma and space
          html = html.substring(0, html.length - 2);
        }
        return html;
      }

      html = '""" + legend_pre + """' + data.xHTML + ':';
      for (var i = 0; i < data.series.length; i++) {  
        var series = data.series[i];
        // Skip not visible series
        if (!series.isVisible) continue;
        if (sepLines) html += '<br>';
        //var decoration = series.isHighlighted ? ' class="highlight"' : '';
        var decoration = series.isHighlighted ? ' style="background-color: #000000"' : '';
        
        /*if (series.isHighlighted) {
            decoration = ' style="background-color: #fcf8b0"'
          }
        else{
            decoration = ' style="background-color: #ffffff"'
        }
        console.log(decoration)*/
        //decoration = ' style="background-color: #fcf8b0"'
        html += "<span" + decoration + "> <b><span style='color: " + series.color + ";'>" + series.labelHTML + "</span></b>:&#160;" + series.yHTML + "</span>, ";
      }
      // Remove last comma and space
      html = html.substring(0, html.length - 2);
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

    global_min, global_max, dg_data = to_dg_data(serie, aggregate_by)
    if global_max != global_min:
        plot_min = str(global_min-((global_max-global_min)*0.1))
        plot_max = str(global_max+((global_max-global_min)*0.1))
    else:
        # Case where there is a straight line
        plot_min = str(global_min*0.99)
        plot_max = str(global_max*1.01)

    dygraphs_javascript += '"Timestamp,{}\\n{}",'.format(labels, dg_data)
    dygraphs_javascript += """{\
drawGrid: true,
drawPoints:"""+drawPoints_value+""",
strokeWidth: 1.5,
pointSize:2.0,
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
    valueRange: [0,1],
    //customBars: false, // Does not work?
  }
},
animatedZooms: true,"""

    if isinstance(serie, DataTimeSlotSerie):
        if aggregate_by:
            rgba_value       = 'rgba(255,128,128,1)' # For the legend
            fill_alpha_value = 0.31                  # For the area
        else:
            rgba_value       = 'rgba(255,128,128,1)' # For the legend
            fill_alpha_value = 0.5                   # For the area
        dygraphs_javascript += """series: {
       'data_loss': {
         //customBars: false, // Does not work?
         axis: 'y2',
         drawPoints: false,
         strokeWidth: 0,
         highlightCircleSize:0,
         fillGraph: true,
         fillAlpha: """+str(fill_alpha_value)+""",            // This alpha is used for the area
         //color: 'rgba(255,0,0,0.6)' // This alpha is used for the legend 
         color: '"""+rgba_value+"""'
       },
     },
"""
    if isinstance(serie, DataTimeSlotSerie) and len(serie[0].data.keys()) <=2:
        dygraphs_javascript += """colors: ['rgb(0,128,128)'],""" # Force "original" Dygraph color.

    if aggregate_by:
        dygraphs_javascript += 'customBars: true'        
    dygraphs_javascript += '})'

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
