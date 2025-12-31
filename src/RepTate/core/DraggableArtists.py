# RepTate: Rheology of Entangled Polymers: Toolkit for the Analysis of Theory and Experiments
# --------------------------------------------------------------------------------------------------------
#
# Authors:
#     Jorge Ramirez, jorge.ramirez@upm.es
#     Victor Boudara, victor.boudara@gmail.com
#
# Useful links:
#     http://blogs.upm.es/compsoftmatter/software/reptate/
#     https://github.com/jorge-ramirez-upm/RepTate
#     http://reptate.readthedocs.io
#
# --------------------------------------------------------------------------------------------------------
#
# Copyright (2017-2023): Jorge Ramirez, Victor Boudara, Universidad Politécnica de Madrid, University of Leeds
#
# This file is part of RepTate.
#
# RepTate is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RepTate is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RepTate.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------------------------------------
"""Module DraggableArtists

Module for the definition of interactive graphical objects that the user can move.

""" 
# draggable matplotlib artists with the animation blit techniques; see
import numpy as np
import enum

class DragType(enum.Enum):
    """Describes the type of drag that the graphical object can be subjected to"""
    vertical = 1
    horizontal = 2
    both = 3
    none = 4
    special = 5

class DraggableArtist(object):
    """Abstract class for motions of a matplotlib artist"""
    lock = None
    def __init__(self, artist=None, mode=DragType.none, function=None, parent_theory=None):
        """**Constructor**"""
        self.parent_theory = parent_theory
        self.artist = artist
        self.press = None
        self.background = None
        self.mode=mode
        self.function=function
        self.data = None
        self.connect()

    def connect(self):
        """Connect events"""
        self.cidpress = self.artist.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.artist.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.artist.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Press events"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.get_data()
        self.press = event.xdata, event.ydata
        DraggableArtist.lock = self
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        axes.draw_artist(self.artist)
        canvas.update()
        #canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Motion event"""
        if DraggableArtist.lock is not self:
            return
        if event.inaxes != self.artist.axes: return
        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        if (self.mode==DragType.none):   
            self.modify_artist(0, 0)
        elif (self.mode==DragType.horizontal):
            self.modify_artist(dx, 0)
        elif (self.mode==DragType.vertical):
            self.modify_artist(0, dy)
        elif (self.mode==DragType.both):
            self.modify_artist(dx, dy)

        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        canvas.restore_region(self.background)
        axes.draw_artist(self.artist)
        # canvas.blit(axes.bbox)
        canvas.update()


    def modify_artist(self, dx, dy):
        """Modify the artist's position or properties.

        Base implementation does nothing. Subclasses override this method
        to implement specific artist modification behavior during drag operations.

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates.
        """
        pass

    def get_data(self):
        """Do nothing"""
        pass

    def on_release(self, event):
        """Release event"""
        if DraggableArtist.lock is not self: return
        xpress, ypress = self.press
        if event.xdata is None: return
        if event.ydata is None: return
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        if (self.mode==DragType.none):   
            self.function(0, 0)
        elif (self.mode==DragType.horizontal):
            self.function(dx, 0)
        elif (self.mode==DragType.vertical):
            self.function(0, dy)
        elif (self.mode==DragType.both):
            self.function(dx, dy)
        self.press = None
        DraggableArtist.lock = None
        self.artist.set_animated(False)
        self.background = None
        self.artist.figure.canvas.draw()
        try:
            self.parent_theory.handle_actionMinimize_Error()
        except AttributeError:
            self.parent_theory.do_fit("")


    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.artist.figure.canvas.mpl_disconnect(self.cidpress)
        self.artist.figure.canvas.mpl_disconnect(self.cidrelease)
        self.artist.figure.canvas.mpl_disconnect(self.cidmotion)

###############################################################
###############################################################


class DraggableBinSeries(DraggableArtist):
    """Dragabble histogram"""
    def __init__(self, artist, mode=DragType.none, logx=False, logy=False, function=None):
        """**Constructor**"""
        super().__init__(artist, mode, function)
        self.logx = logx
        self.logy = logy
    
    def on_press(self, event):
        """Press event"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return 
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.xdata, self.ydata = self.artist.get_data()
        nmodes=len(self.xdata)
        try:
            auxshape = self.xdata.shape[1]
        except IndexError:
            auxshape = 0
        if auxshape>1:
            self.xdata = self.xdata[:,0]
            self.ydata = self.ydata[:,0]
        self.xdata_at_press = self.xdata
        self.ydata_at_press = self.ydata
        self.press = event.xdata, event.ydata
        # Index of mode clicked
        self.index = np.argmin((self.xdata-self.press[0])**2+(self.ydata-self.press[1])**2)
        DraggableArtist.lock = self
        # draw everything but the selected curve and store in 'background'
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        # redraw just the curve
        axes.draw_artist(self.artist)

    def on_motion(self, event):
        """Motion event"""
        if DraggableArtist.lock is not self:
            return
        if event.inaxes != self.artist.axes:
            return
        self.xpress, self.ypress = self.press
        if self.logx:
            dx = np.log10(event.xdata) - np.log10(self.xpress)
        else:
            dx = event.xdata - self.xpress
        if self.logy:
            dy = np.log10(event.ydata) - np.log10(self.ypress)
        else:
            dy = event.ydata - self.ypress

        self.modify_artist(dx, dy)        
        
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        # restore the background
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        canvas.update()

    def modify_artist(self, dx, dy):
        """Update histogram bin position during drag.

        Modifies the x-coordinate of the selected bin based on displacement,
        respecting logarithmic scaling if enabled. Updates the artist's data
        to reflect the new position.

        Args:
            dx (float): Horizontal displacement. In linear scale, this is a direct
                offset. In log scale, this is a logarithmic displacement (log10).
            dy (float): Vertical displacement. In linear scale, this is a direct
                offset. In log scale, this is a logarithmic displacement (log10).
        """
        xdata = self.xdata_at_press
        ydata = self.ydata_at_press
        xdataind = xdata[self.index]
        ydataind = ydata[self.index]
        nmodes = len(self.xdata)
        if self.logx:
            newx = self.xpress*np.power(10, dx)
        else:
            newx = self.xpress + dx
        if self.logy:
            newy = self.ypress*np.power(10, dy)
        else:
            newy = self.ypress + dy

        newxdata=xdata
        newydata=ydata
        # if self.index==0:
        #     newxdata[0] = newx
        #     newydata[0] = newy
        #     newxdata = np.linspace(newx, newxdata[nmodes-1], nmodes)
        #     newxdata=newxdata.reshape(nmodes,1)
        # elif self.index==nmodes-1:
        #     newxdata[self.index] = newx
        #     newydata[self.index] = newy
        #     newxdata = np.linspace(newxdata[0], newx, nmodes)
        #     newxdata=newxdata.reshape(nmodes,1)
        # else:
        newxdata[self.index] = newx
        # newydata[self.index] = newy


        self.artist.set_data(newxdata, newydata)

    def on_release(self, event):
        """Release event"""
        if DraggableArtist.lock is not self: return
        xpress, ypress = self.press
        if event.xdata is None: return
        if event.ydata is None: return

        #dx = event.xdata - xpress
        #dy = event.ydata - ypress
        #if (self.mode==DragType.none):   
        #    self.function(0, 0)
        #elif (self.mode==DragType.horizontal):
        #    self.function(dx, 0)
        #elif (self.mode==DragType.vertical):
        #    self.function(0, dy)
        #elif (self.mode==DragType.both):
        #    self.function(dx, dy)
        self.press = None
        DraggableArtist.lock = None
        self.artist.set_animated(False)
        # restore the background
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        #update
        # canvas.update()
        # canvas.blit(axes.bbox)
        self.background = None
        # self.artist.figure.canvas.draw()
        self.data = self.artist.get_data()
        xdata = self.data[0]
        ydata = self.data[1]
        self.function(xdata, ydata)

################################################################        
################################################################        


class DraggableModesSeries(DraggableArtist):
    """Draggable points of a series"""
    def __init__(self, artist, mode=DragType.none, parent_application=None, function=None):
        """**Constructor**"""
        super(DraggableModesSeries, self).__init__(artist, mode, function)
        self.parent_application = parent_application
        self.update_logx_logy()
    
    def update_logx_logy(self):
        """Update logarithmic scale flags from parent application view.

        Synchronizes the logx and logy attributes with the current view's
        logarithmic scale settings. This ensures drag operations correctly
        handle coordinate transformations in log-scaled plots.
        """
        self.logx = self.parent_application.current_view.log_x
        self.logy = self.parent_application.current_view.log_y

    def on_press(self, event):
        """Press event"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.xdata, self.ydata = self.artist.get_data()
        nmodes=len(self.xdata)
        auxshape = self.xdata.shape[1]
        if auxshape>1:
            self.xdata = self.xdata[:,0]
            self.ydata = self.ydata[:,0]
        self.xdata_at_press = self.xdata
        self.ydata_at_press = self.ydata
        self.press = event.xdata, event.ydata
        # Index of mode clicked
        self.index = np.argmin((self.xdata-self.press[0])**2+(self.ydata-self.press[1])**2)
        DraggableArtist.lock = self
        # draw everything but the selected curve and store in 'background'
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        # redraw just the curve
        axes.draw_artist(self.artist)
        #canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Motion event"""
        if DraggableArtist.lock is not self:
            return
        if event.inaxes != self.artist.axes: return
        self.xpress, self.ypress = self.press
        self.update_logx_logy()
        if self.logx:
            dx = np.log10(event.xdata) - np.log10(self.xpress)
        else:
            dx = event.xdata - self.xpress
        if self.logy:
            dy = np.log10(event.ydata) - np.log10(self.ypress)
        else:
            dy = event.ydata - self.ypress

        self.modify_artist(dx, dy)        
        
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        # restore the background
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        canvas.update()

    def modify_artist(self, dx, dy):
        """Update mode series coordinates during drag.

        Modifies the selected point's coordinates with special handling for
        boundary modes (first and last). When dragging the first or last mode,
        all x-coordinates are redistributed linearly or logarithmically between
        the new position and the opposite boundary. Y-coordinates of the selected
        mode are always updated.

        Args:
            dx (float): Horizontal displacement. In linear scale, this is a direct
                offset. In log scale, this is a logarithmic displacement (log10).
            dy (float): Vertical displacement. In linear scale, this is a direct
                offset. In log scale, this is a logarithmic displacement (log10).
        """
        xdata = self.xdata_at_press
        ydata = self.ydata_at_press
        xdataind = xdata[self.index]
        ydataind = ydata[self.index]
        nmodes = len(self.xdata)
        self.update_logx_logy()
        if self.logx:
            newx = self.xpress*np.power(10, dx)
        else:
            newx = self.xpress + dx
        if self.logy:
            newy = self.ypress*np.power(10, dy)
        else:
            newy = self.ypress + dy

        newxdata=xdata
        newydata=ydata
        if self.index==0:
            if self.logx:
                newxdata = np.power(10, np.linspace(np.log10(newx), np.log10(newxdata[nmodes-1]), nmodes))
            else:
                newxdata = np.linspace(newx, newxdata[nmodes-1], nmodes)
            newxdata = newxdata.reshape(nmodes,1)
        elif self.index==nmodes-1:
            if self.logy:
                newxdata = np.power(10, np.linspace(np.log10(newxdata[0]), np.log10(newx), nmodes))
            else:
                newxdata = np.linspace(newxdata[0], newx, nmodes)
            newxdata=newxdata.reshape(nmodes,1)

        newydata[self.index] = newy

        self.artist.set_data(newxdata, newydata)

    def on_release(self, event):
        """Release event"""
        if DraggableArtist.lock is not self: return
        xpress, ypress = self.press
        if event.xdata is None: return
        if event.ydata is None: return

        #dx = event.xdata - xpress
        #dy = event.ydata - ypress
        #if (self.mode==DragType.none):   
        #    self.function(0, 0)
        #elif (self.mode==DragType.horizontal):
        #    self.function(dx, 0)
        #elif (self.mode==DragType.vertical):
        #    self.function(0, dy)
        #elif (self.mode==DragType.both):
        #    self.function(dx, dy)
        self.press = None
        DraggableArtist.lock = None
        self.artist.set_animated(False)
        # restore the background
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        #update
        # canvas.update()
        # canvas.blit(axes.bbox)
        self.background = None
        # self.artist.figure.canvas.draw()
        tmp_data = self.artist.get_data()
        xdata = tmp_data[0]
        ydata = tmp_data[1]
        # for compatibility with mpldatacursor
        # prevent the modes from disappearing 
        try:
            float(xdata[0])
            self.data = tmp_data
        except TypeError:
            # don't change the modes
            xdata = self.data[0]
            ydata = self.data[1]
        self.function(xdata, ydata)


###########################################################
###########################################################

class DraggableSeries(DraggableArtist):
    """Full draggable series"""
    def __init__(self, artist, mode=DragType.none, logx=False, logy=False, xref=0, yref=0, function=None, functionendshift=None, index=0):
        """**Constructor**"""
        super(DraggableSeries, self).__init__(artist, mode, function)
        self.logx = logx
        self.logy = logy
        self.xref = xref
        self.yref = yref
        self.functionendshift = functionendshift
        self.index = index

        self.dx = 0
        self.dy = 0

    def get_data(self):
        """Return data"""
        self.data = self.artist.get_data()
    
    def on_press(self, event):
        """Press event"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.press = event.xdata, event.ydata
        self.get_data()
        DraggableArtist.lock = self
        # draw everything but the selected curve and store in 'background'
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        # redraw just the curve
        axes.draw_artist(self.artist)
        #canvas.blit(axes.bbox)

    def on_motion(self, event):
        """Motion event"""
        if DraggableArtist.lock is not self:
            return
        if event.inaxes != self.artist.axes: return
        xpress, ypress = self.press
        if self.logx:
            self.dx = np.log10(event.xdata) - np.log10(xpress)
        else:
            self.dx = event.xdata - xpress
        if self.logy:
            self.dy = np.log10(event.ydata) - np.log10(ypress)
        else:
            self.dy = event.ydata - ypress

        if (self.mode==DragType.none):   
            self.modify_artist(0, 0)
            self.function(0, 0, self.index)
        elif (self.mode==DragType.horizontal):
            self.modify_artist(self.dx, 0)
            self.function(self.dx, 0, self.index)
        elif (self.mode==DragType.vertical):
            self.modify_artist(0, self.dy)
            self.function(0, self.dy, self.index)
        elif (self.mode==DragType.both):
            self.modify_artist(self.dx, self.dy)
            self.function(self.dx, self.dy, self.index)
        
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        # restore the background
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        canvas.update()

    def modify_artist(self, dx, dy):
        """Update entire series coordinates during drag.

        Applies the displacement to all points in the series, respecting
        logarithmic scaling if enabled. This allows the entire data series
        to be shifted uniformly in x, y, or both directions.

        Args:
            dx (float): Horizontal displacement. In linear scale, this is a direct
                offset added to all x-coordinates. In log scale, this is a logarithmic
                displacement (log10) used as a multiplicative factor.
            dy (float): Vertical displacement. In linear scale, this is a direct
                offset added to all y-coordinates. In log scale, this is a logarithmic
                displacement (log10) used as a multiplicative factor.
        """
        if self.logx:
            newx = [x*np.power(10, dx) for x in self.data[0]]
        else:
            newx = [x + dx for x in self.data[0]]
        if self.logy:
            newy = [y*np.power(10, dy) for y in self.data[1]]
        else:
            newy = [y + dy for y in self.data[1]]
        self.artist.set_data(newx, newy)

    def on_release(self, event):
        """Release event"""
        if DraggableArtist.lock is not self: return
        if (self.mode==DragType.none):   
            self.functionendshift(0, 0, self.index)
        elif (self.mode==DragType.horizontal):
            self.functionendshift(self.dx, 0, self.index)
        elif (self.mode==DragType.vertical):
            self.functionendshift(0, self.dy, self.index)
        elif (self.mode==DragType.both):
            self.functionendshift(self.dx, self.dy, self.index)
        self.press = None
        DraggableArtist.lock = None
        self.artist.set_animated(False)
        # restore the background
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        canvas.restore_region(self.background)
        # draw the curve only
        axes.draw_artist(self.artist)
        #update
        # canvas.update()
        # canvas.blit(axes.bbox)
        self.background = None
        # self.artist.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        super().disconnect()

class DraggablePatch(DraggableArtist):
    """Draggable Patch"""
    def __init__(self, artist, mode=DragType.none, function=None):
        """**Constructor**"""
        super(DraggablePatch, self).__init__(artist, mode, function)

    def get_data(self):
        """Get data of the artist"""
        self.data=self.artist.center

    def modify_artist(self, dx, dy):
        """Update patch center position during drag.

        Translates the patch by adjusting its center coordinates based on
        the provided displacement values.

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates.
        """
        self.artist.center = (self.data[0]+dx, self.data[1]+dy)

class DraggableRectangle(DraggableArtist):
    """Draggable rectangle"""
    def __init__(self, artist, mode=DragType.none, function=None):
        """**Constructor**"""
        super(DraggableRectangle, self).__init__(artist, mode, function)

    def get_data(self):
        """Get data of the artist"""
        self.data=self.artist.xy

    def modify_artist(self, dx, dy):
        """Update rectangle position during drag.

        Translates the rectangle by adjusting its bottom-left corner (xy)
        coordinates based on the provided displacement values.

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates.
        """
        self.artist.set_x(self.data[0]+dx)
        self.artist.set_y(self.data[1]+dy)

class DraggableVLine(DraggableArtist):
    """Draggable Verticla line"""
    def __init__(self, artist, mode=DragType.none, function=None, parent_theory=None):
        """**Constructor**"""
        super(DraggableVLine, self).__init__(artist, mode, function, parent_theory)
    
    def on_press(self, event):
        """Press event"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.get_data()
        self.press = self.data[0][0], 0 # do not use event.xdata, precision matters in non-logscale
        DraggableArtist.lock = self
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        axes.draw_artist(self.artist)
        canvas.update()
        #canvas.blit(axes.bbox)

    def get_data(self):
        """Get the data from the artist"""
        self.data = self.artist.get_data()

    def modify_artist(self, dx, dy):
        """Update vertical line position during drag.

        Shifts the vertical line horizontally by the displacement. The vertical
        extent remains unchanged (normalized to [0, 1] in axes coordinates).

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates (ignored for vertical lines).
        """
        self.artist.set_data([self.data[0][0] + dx, self.data[0][1] + dx], [0, 1])


class DraggableHLine(DraggableArtist):
    """Draggable Horizontal line"""
    def __init__(self, artist, mode=DragType.none, function=None, parent_theory=None):
        """**Constructor**"""
        super(DraggableHLine, self).__init__(artist, mode, function, parent_theory)
    
    def on_press(self, event):
        """Press event"""
        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.get_data()
        self.press = 0, self.data[1][0] # do not use event.ydata, precision matters in non-logscale
        DraggableArtist.lock = self
        canvas = self.artist.figure.canvas
        axes = self.artist.axes
        self.artist.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.artist.axes.bbox)
        axes.draw_artist(self.artist)
        canvas.update()
        #canvas.blit(axes.bbox)
   
    def get_data(self):
        """Get the artist data"""
        self.data = self.artist.get_data()

    def modify_artist(self, dx, dy):
        """Update horizontal line position during drag.

        Shifts the horizontal line vertically by the displacement. The horizontal
        extent remains unchanged (normalized to [0, 1] in axes coordinates).

        Args:
            dx (float): Horizontal displacement in data coordinates (ignored for horizontal lines).
            dy (float): Vertical displacement in data coordinates.
        """
        self.artist.set_data([0, 1], [self.data[1][0] + dy, self.data[1][1] + dy])

class DraggableVSpan(DraggableArtist):
    """Draggable Vertical Span"""
    def __init__(self, artist, mode=DragType.none, function=None):
        """**Constructor**"""
        super(DraggableVSpan, self).__init__(artist, mode, function)

    def get_data(self):
        """Get the artist data"""
        self.data=self.artist.get_xy()

    def modify_artist(self, dx, dy):
        """Update vertical span position during drag.

        Shifts the vertical span horizontally by the displacement while
        preserving its width. The vertical extent remains unchanged
        (normalized to [0, 1] in axes coordinates).

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates (ignored for vertical spans).
        """
        xmin = self.data[0][0]
        xmax = self.data[2][0]
        self.artist.set_xy([[xmin+dx,0],[xmin+dx,1],[xmax+dx,1],[xmax+dx,0],[xmin+dx,0]])

class DraggableHSpan(DraggableArtist):
    """Draggable Horizontal Span"""
    def __init__(self, artist, mode=DragType.none, function=None):
        """**Constructor**"""
        super(DraggableHSpan, self).__init__(artist, mode, function)

    def get_data(self):
        """Get the artist data"""
        self.data=self.artist.get_xy()

    def modify_artist(self, dx, dy):
        """Update horizontal span position during drag.

        Shifts the horizontal span vertically by the displacement while
        preserving its height. The horizontal extent remains unchanged
        (normalized to [0, 1] in axes coordinates).

        Args:
            dx (float): Horizontal displacement in data coordinates (ignored for horizontal spans).
            dy (float): Vertical displacement in data coordinates.
        """
        ymin = self.data[0][1]
        ymax = self.data[1][1]
        self.artist.set_xy([[0, ymin+dy], [0, ymax+dy], [1, ymax+dy], [1 ,ymin+dy], [0, ymin+dy]])

class DraggableNote(DraggableArtist):
    """Draggable annotation box"""
    def __init__(self, artist, mode=DragType.none, function=None, function2=None):
        """**Constructor**"""
        super(DraggableNote, self).__init__(artist, mode, function)
        self.cidpress = self.artist.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.function2=function2

    def get_data(self):
        """Get the artist data"""
        self.data=self.artist.get_position()

    def modify_artist(self, dx, dy):
        """Update annotation box position during drag.

        Relocates the annotation by setting its position based on the original
        press coordinates plus the displacement. This allows free movement of
        the annotation box across the plot.

        Args:
            dx (float): Horizontal displacement in data coordinates.
            dy (float): Vertical displacement in data coordinates.
        """
        self.artist.set_position([self.press[0]+dx, self.press[1]+dy])

    def on_press(self, event):
        """Press event"""
        if not event.dblclick:
            super(DraggableNote, self).on_press(event)
            return

        if event.inaxes != self.artist.axes: return
        if DraggableArtist.lock is not None: return
        if event.button != 1: return
        contains, attrd = self.artist.contains(event)
        if not contains: return
        self.function2(self.artist)
                    
    def on_release(self, event):
        """Release event"""
        if DraggableArtist.lock is not self: return
        xpress, ypress = self.press
        if event.xdata is None: return
        if event.ydata is None: return
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.press = None
        DraggableArtist.lock = None
        self.artist.set_animated(False)
        self.background = None
        self.artist.figure.canvas.draw()
