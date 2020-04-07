import os
import uuid
import datetime
from .time import dt_from_s
from .datastructures import DataTimePointSerie

# Setup logging
import logging
logger = logging.getLogger(__name__)

HARD_DEBUG = False


# Utitlities
def to_dg_time(dt):
    '''Get Dygraphs time form datetime'''
    return '{}/{:02d}/{:02d} {:02d}:{:02d}:{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute,dt.second)


def to_dg_data(dataTimePointSerie, aggregate_by=0):
    '''{% for timestamp,value in metric_data.items %}{{timestamp}},{{value}}\n{%endfor%}}'''
    
    dg_data=''
    data_sum = 0
    data_min = None
    data_max = None
    first_t  = None
    last_t   = None
    
    for i, dataTimePoint in enumerate(dataTimePointSerie):
        if aggregate_by:

            if first_t is None:
                first_t = dataTimePoint.t
                
            # Sum
            data_sum += dataTimePoint.data
            
            # Min
            if data_min is None:
                data_min = dataTimePoint.data
            else:
                if dataTimePoint.data < data_min:
                    data_min = dataTimePoint.data

            # Max
            if data_max is None:
                data_max = dataTimePoint.data
            else:
                if dataTimePoint.data > data_max:
                    data_max = dataTimePoint.data

            if (i!=0)  and ((i % aggregate_by)==0):
                last_t = dataTimePoint.t
                t = first_t + ((last_t-first_t) /2)
                dg_data+='{},{};{};{}\\n'.format(to_dg_time(dt_from_s(t)), data_min, data_sum/aggregate_by, data_max)
                data_sum = 0
                data_min = None
                data_max = None

        else:
            dg_data+='{},{}\\n'.format(to_dg_time(dt_from_s(dataTimePoint.t)), dataTimePoint.data)

    return dg_data



def dygraphs_plot(serie):
    '''Plot a dataTimePointSeries in Jupyter using Dugraph. Based on the work here: https://www.stefaanlippens.net/jupyter-custom-d3-visualization.html'''
    from IPython.display import display, Javascript, HTML

    # Checks
    if isinstance(serie, DataTimePointSerie):
        pass
    else:
        raise Exception('Don\'t know how to plot an object of type "{}"'.format(serie.__class__.__name__))

    # Set label
    if serie.label:
        label = serie.label
    else:
        label = 'value'
    
    # Set title    
    title = serie.__class__.__name__

    aggregate_by = serie.plot_aggregate_by
    logger.info('Aggregating by "{}" for plotting'.format(aggregate_by))
            
    # Define div name
    div_uuid = str(uuid.uuid4())
    graph_div_id  = div_uuid + '_graph'
    legend_div_id = div_uuid + '_legend'


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

      html = data.xHTML + ':';
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
    dygraphs_javascript += '"Timestamp,{}\\n{}",'.format(label, to_dg_data(serie, aggregate_by))
    dygraphs_javascript += """{\
drawGrid: true,
drawPoints:true,
strokeWidth: 1.5,
pointSize:2.0,
highlightCircleSize:4,
stepPlot: false,
fillGraph: false,
fillAlpha: 0.5,
colorValue: 0.5,
showRangeSelector: true,
hideOverlayOnMouseOut: true,
interactionModel: Dygraph.defaultInteractionModel,
includeZero: false,
showRoller: false,
legend: 'always',
legendFormatter: legendFormatter,
labelsDiv: '"""+legend_div_id+"""',
animatedZooms: true,"""
    if aggregate_by:
        dygraphs_javascript += 'customBars: true'        
    dygraphs_javascript += '})'
     
     
    rendering_javascript = '''
require.undef('timseriaplot'); // Helps to reload in Jupyter
define('timseriaplot', ['dgenv'], function (Dygraph) {
    function draw(div_id) {
    ''' + dygraphs_javascript +'''
    }
    return draw;
});
'''

    STATIC_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/static/'

    # Require Dygraphs Javascript library
    # https://raw.githubusercontent.com/sarusso/Timeseria/develop/timeseria/static/js/dygraph-2.1.0.min
    display(Javascript("require.config({paths: {dgenv: '//cdnjs.cloudflare.com/ajax/libs/dygraph/2.1.0/dygraph.min'}});"))
    # TODO: load it locally, maybe with something like:    
    #with open(STATIC_DATA_PATH+'js/dygraph-2.1.0.min.js') as dy_js_file:
    #    display(Javascript(dy_js_file.read()))

    # Load Dygraphs CSS
    with open(STATIC_DATA_PATH+'css/dygraph-2.1.0.css') as dg_css_file:
        display(HTML('<style>'+dg_css_file.read()+'</style>'))
    
    # Load Dygraphs Javascript and html code    
    display(Javascript(rendering_javascript))
    display(HTML('<div style="height:45px; margin-left:0px"><div id="{}" style="width:100%"></div></div>'.format(legend_div_id)))
    display(HTML('<div id="{}" style="width:100%; margin-right:0px"></div>'.format(graph_div_id)))

    # Render the graph (this is the "entrypoint")
    display(Javascript("""
        (function(element){
            require(['timseriaplot'], function(timseriaplot) {
                timseriaplot(element.get(0), '%s');
            });
        })(element);
    """ % (graph_div_id)))



def matplotlib_plot(serie):
    def as_vectors(self):
        X = [datetime.datetime.fromtimestamp(item.t) for item in serie]
        Y = [item.data for item in serie]
        return X,Y
    from matplotlib import pyplot
    pyplot.plot(*as_vectors(serie))
    pyplot.show()
