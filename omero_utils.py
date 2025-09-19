import dask.array as da
from dask import delayed
import dask
import numpy as np
import time
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import simpledialog

from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.model import ProjectI, DatasetI, ImageI, ProjectDatasetLinkI, MapAnnotationI 
from omero.model import RoiI, PolygonI, RectangleI, MaskI, LengthI
from omero.rtypes import unwrap, rstring, rlong, robject
from omero.sys import ParametersI
from omero.constants.metadata import NSCLIENTMAPANNOTATION
from omero.model.enums import UnitsLength

import omero_rois

# For more information see: https://docs.openmicroscopy.org/omero/5.4.0/developers/Python.html


def connect (username, host = "omero-prod.camp.thecrick.org") :
    """
    Connect to Crick OMERO server.

    Parameters:
        username: Crick username
        host (str): address for OMERO host. For default Crick host, on-site or active VPN requied for access.
    Returns:
        Connected BlitzGateway
    """

    def get_password ():
        """
        Secure method for inputting password.
        
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.lift()            # Lift it above all windows
        root.attributes("-topmost", True)   # Make sure it stays on top
        root.after_idle(root.attributes, "-topmost", False)  # Reset "always on top"
            
        # Retrieve password securely
        password = simpledialog.askstring("Password", "Enter your password:", show="*")
        return password

    PASSWORD = get_password()

    USERNAME = username
    OMERO_HOST = host 
    PORT = 4064

    conn = BlitzGateway(USERNAME, PASSWORD, host=OMERO_HOST, port=PORT, secure=True)
    connected = conn.connect()
    conn.c.enableKeepAlive(60)
    
    if not connected:
        print("Could not connect to OMERO server.")
    
    else:
        print(f"Connected to OMERO server: {OMERO_HOST}")

    return conn

def set_group (conn, group_id):
    """
    Set OMERO group.

    Parameters:
        conn: Connected BlitzGateway.
        group_id (int): OMERO group ID.
    """
    conn.SERVICE_OPTS.setOmeroGroup(group_id)
    g = conn.SERVICE_OPTS.getOmeroGroup()
    print(f"Group set to {g}") 

def print_obj(obj, indent=0):
    """
    Helper method to display info about OMERO objects.
    Not all objects will have a "name" or owner field.
    """
    print("""%s%s:%s  Name:"%s" (owner=%s)""" % (
        " " * indent,
        obj.OMERO_CLASS,
        obj.getId(),
        obj.getName(),
        obj.getOwnerOmeName()))

def list_projects (conn):
    for prj in conn.getObjects("Project"):
        #print(f"project name: {prj.getName()}  ID: {prj.getId()}")
        print_obj(prj)
    return None
    
def get_image (conn, image_id):
    img = conn.getObject("Image", image_id)
    #print(f"image name: {img.getName()} \nimage ID: {img.getId()}")
    print_obj(img)
    return img

def get_dataset (conn, dataset_id):
    dat = conn.getObject("Dataset", dataset_id)
    #print(f"dataset name: {dat.getName()} \ndataset ID: {dat.getId()}")
    print_obj(dat)
    if dat.countChildren() > 0:
        children = dat.listChildren()
        for img in children:
            #print(f"  image name: {img.getName()} \n  image ID: {img.getId()}")
            print_obj(img, indent=1)
    else :
        print(f"dataset does not contain any images")
    return dat

def get_project (conn, project_id):
    prj = conn.getObject("Project", project_id)
    #print(f"project name: {prj.getName()} \nproject ID: {prj.getId()}")
    print_obj(prj)
    if prj.countChildren() > 0:
        children = prj.listChildren()
        for dat in children:
            #print(f"  dataset name: {dat.getName()} \n  dataset ID: {dat.getId()}")
            print_obj(dat, indent=1)
            children2 = dat.listChildren()
            for img in children2:
                #print(f"    image name: {img.getName()} \n    image ID: {img.getId()}")
                print_obj(img, indent=2)
    else :
        print(f"project does not contain any images")
    return prj

def get_plane(plane_name, primaryPixels, cache=None): # plane_name = (z, c, t) coordinates, pixels = primary pixels object for image ID
    """ 
    Loads a single plane into cache.
    Parameters:
        plane_name (list): [z, c, t] coordinates
        primaryPixels: OMERO primary pixel object for image ID
        cache (dict, optional): optional dictionary to cache loaded planes

    Returns:
        2D array of pixel values
    
    """
    
    z, c, t = plane_name
    print(f"Loading plane z={z}, c={c}, t={t}")

    start_time = time.time()

    if cache is not None:
        if (z, c, t) not in cache:
            cache[(z, c, t)] = np.array(primaryPixels.getPlane(z, c, t))  # Ensure NumPy array
        plane = cache[(z, c, t)]  
    else :
        plane = np.array(primaryPixels.getPlane(z, c, t))  # Ensure NumPy array
        
    end_time = time.time()
    print(f"Time to load plane {plane_name} = {end_time - start_time:.2f} seconds")

    return plane

# Function to lazily load a plane
def get_planes(zct_list, pixels):
    """Load multiple planes using OMERO getPlanes"""
    #start_time = time.time()
    
    planes = pixels.getPlanes(zct_list)  # This is a generator, loading only when needed
    
    #end_time = time.time()
    #print(f"Time to load stack = {end_time - start_time:.2f} seconds")
    return np.array([plane for plane in planes])  # Ensure correct shape

def get_lazy_stack(img, cache=None):
    """
    Create a lazy stack for an OMERO image.

    Parameters:
    """
    print(img.name)

    primaryPixels = img.getPrimaryPixels()
    
    # Get metadata
    sizeX, sizeY, sizeZ, sizeT, sizeC = img.getSizeX(), img.getSizeY(), img.getSizeZ(), img.getSizeT(), img.getSizeC()
    
    # Ensure dtype is valid
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}
    pixelsType = dtype_map.get(img.getPixelsType(), np.uint16)  # Default to uint16

    # Create lazy stack
    lazy_arrays = [
        da.from_delayed(
            delayed(get_plane)([z, c, t], primaryPixels, cache), 
            shape=(sizeY, sizeX), 
            dtype=pixelsType
        ) 
        for c in range(sizeC)
        for t in range(sizeT)
        for z in range(sizeZ)
    ]

    # Stack and reshape
    lazy_stack = da.stack(lazy_arrays, axis=0).reshape((sizeC, sizeT, sizeZ, sizeY, sizeX))
    
    return lazy_stack

def mask_to_array(mask):
    if isinstance(mask, MaskI) :
        x = int(mask.getX().getValue())
        y = int(mask.getY().getValue())
        w = int(mask.getWidth().getValue())
        h = int(mask.getHeight().getValue())
    
        mask_packed = mask.getBytes()
        # convert bytearray into something we can use
        intarray = np.frombuffer(mask_packed, dtype=np.uint8)
        binarray = np.unpackbits(intarray).astype(np.uint8)
        # truncate and reshape
        binarray = np.reshape(binarray[: (w * h)], (h, w))
    
        return binarray, (y, x, h, w)
    else :
        print(f"skipping non-MaskI object: {type(mask)}")
        return [], (0,0,0,0)

def find_images_in_project_by_hql(conn, project_id, key, value):
    key = str(key)
    value = str(value)
    query_service = conn.getQueryService()
    
    # HQL query
    hql_query = """
    SELECT image
    FROM Project AS project
    JOIN project.datasetLinks AS dataset_links
    JOIN dataset_links.child AS dataset
    JOIN dataset.imageLinks AS image_links
    JOIN image_links.child AS image
    JOIN image.annotationLinks AS annotation_links
    JOIN annotation_links.child AS annotation
    JOIN annotation.mapValue as map_value
    WHERE TYPE(annotation) = MapAnnotation
    AND project.id = :id
    AND map_value.name = :key
    AND map_value.value = :value
    """

    # Parameters
    params = ParametersI()
    params.addId(rlong(project_id))
    params.addString(f"key", rstring(key))
    params.addString(f"value", rstring(value))

    results = query_service.findAllByQuery(hql_query, params)

    return results

def find_image_ids_in_project_kv_dict (conn, project_id, kv_dict):
    list_of_results = []
    for i, (k, v) in enumerate(kv_dict.items()):
        results = find_images_in_project_by_hql(conn, project_id, k, v)
        image_ids = [result.id.val for result in results]
        
        list_of_results.append(set(image_ids))
    common_values = list(set.intersection(*list_of_results))
    return common_values

def find_image_names_in_project_kv_dict (conn, project_id, kv_dict):
    list_of_results = []
    for i, (k, v) in enumerate(kv_dict.items()):
        results = find_images_in_project_by_hql(conn, project_id, k, v)
        image_names = [result.name.val for result in results]
        
        list_of_results.append(set(image_names))
    common_values = list(set.intersection(*list_of_results))
    return common_values

def create_project(conn, project_name):
    """ 
    Create omero project.
    Checks if a project with project_name exists.
    
    
    Parameters:
        conn: Connected BlitzGateway.
        project_name (str): project name.
    
    Returns:
        OMERO project object.
    
    """
    # Check is project exists
    project_exists = None
    for proj in conn.getObjects("Project"):
        if proj.getName() == str(project_name):
            project_exists = proj
            break

    if project_exists:
        print(f"Project already exists: ID = {project_exists.getId()}, Name = {project_exists.getName()}")
        project_obj = project_exists
        project_id = project_obj.getId()
        project_name = project_obj.getName()

    else:
        # Create a new project
        project_obj = ProjectI()
        project_obj.setName(rstring(project_name))
        project_obj = conn.getUpdateService().saveAndReturnObject(project_obj)
        project_id = project_obj.getId().getValue()
        project_name = project_obj.getName().getValue()
        print(f"Created new project: ID = {project_id}, Name = {project_name}")

    # Re-load project object
    project_obj = conn.getObject("Project", project_id)
    return project_obj

def create_dataset(conn, dataset_name, project_id):
    """ 
    Create omero dataset within defined project.
    Checks if a dataset with dataset_name exists.
    Creates and returns new dataset object or returns existing.
    
    Parameters:
        conn: Connected BlitzGateway.
        dataset_name (str): dataset name.
        project_id (int): 
        
    Returns:
        OMERO dataset object.
    
    """
    # Search for existing project with the same name
    dataset_exists = None
    project_obj = conn.getObject("Project", project_id)
    if project_obj.countChildren() > 0:
        for dataset in project_obj.listChildren():
            if dataset.getName() == str(dataset_name):
                dataset_exists = dataset
                break 

    if dataset_exists:
        print(f"Dataset already exists: ID = {dataset_exists.getId()}, Name = {dataset_exists.getName()}")
        dataset_obj = dataset_exists
        dataset_id = dataset_obj.getId()
        dataset_name = dataset_obj.getName()

    else:
        # Create a new dataset
        dataset_obj = DatasetI()
        dataset_obj.setName(rstring(dataset_name))
        dataset_obj = conn.getUpdateService().saveAndReturnObject(dataset_obj)
        dataset_id = dataset_obj.getId().getValue()
        dataset_name = dataset_obj.getName().getValue()
        print(f"Created new dataset: ID = {dataset_id}, Name = {dataset_name}")

        # link dataset to project
        link = ProjectDatasetLinkI()
        link.setParent(ProjectI(project_obj.getId(), False))
        link.setChild(dataset_obj)
        conn.getUpdateService().saveObject(link)

    # Re-load dataset object
    dataset_obj = conn.getObject("Dataset", dataset_id)
    return dataset_obj

def create_image(conn, image_name, dataset_id, key_value_pairs, image_planes, channel_names=None, description=None, 
                 sizeZ=1, sizeC=1, sizeT=1, pixel_size_um = 0, 
                 sourceImageId=None, channelList=None):
    """ 
    Create omero image within defined dataset.
    Checks existing image with image_name.
    Creates and returns new image object or returns existing.
    
    Parameters:
        conn: Connected BlitzGateway.
        image_name (str): name of new image.
        dataset_id (int): link image to this dataset
        key_value_pairs (list): List of metadata [key, value] pairs for MapAnnotations
        image_planes (numpy_ndarray): image planes in numpy sequence
        description (str): description for new image
        
    Returns:
        OMERO image object.
    
    """
    # Search for existing project with the same name
    image_exists = None
    dataset_obj = conn.getObject("Dataset", dataset_id)
    if dataset_obj.countChildren() > 0:
        for image in dataset_obj.listChildren():
            if image.getName() == str(image_name):
                image_exists = image
                break 

    if image_exists:
        print(f"Image already exists: ID = {image_exists.getId()}, Name = {image_exists.getName()}")
        image_obj = image_exists
        image_id = image_obj.getId()
        image_name = image_obj.getName()

    else:
        # Stack into a 3D array
        image_stack = np.stack(image_planes, axis=0)
        
        # Create a new image
        def plane_gen():
            """generator will yield planes"""
            for plane in image_stack:
                yield plane
                
        img_obj = conn.createImageFromNumpySeq(
                plane_gen(), image_name, sizeZ, sizeC, sizeT, description=description, 
                dataset=dataset_obj, sourceImageId=sourceImageId, channelList=channelList
            )
        
        image_id = img_obj.getId()#.getValue()
        image_name = img_obj.getName()#.getValue()
        print(f"Created new image: ID = {image_id}, Name = {image_name}")

        # re-load image object to avoid conflicts
        img_obj = conn.getObject("Image", image_id)

        # set channel names
        if (channel_names != None):
            channel_labels = dict(enumerate(channel_names, start=1))
            conn.setChannelNames('Image', [img_obj.getId()], 
                                 channel_labels
                                )
        
        # set pixel sizes
        if pixel_size_um <= 0:
            px_size_um = LengthI(1, UnitsLength.PIXEL) # non calibrated pixel sizes
        else :
            px_size_um = LengthI(pixel_size_um, UnitsLength.MICROMETER)
        px_obj = img_obj.getPrimaryPixels()._obj
        px_obj.setPhysicalSizeX(px_size_um)
        px_obj.setPhysicalSizeY(px_size_um)
        conn.getUpdateService().saveObject(px_obj)

        # Re-load the image to avoid update conflicts
        img_obj = conn.getObject("Image", img_obj.getId())
        
        map_ann = MapAnnotationWrapper(conn)
        map_ann.setNs(NSCLIENTMAPANNOTATION)
        map_ann.setValue(key_value_pairs)
        map_ann.save()
        
        # NB: only link a client map annotation to a single object
        map_ann = img_obj.linkAnnotation(map_ann)

    # Re-load image object to avoid conflicts
    image_obj = conn.getObject("Image", image_id)
    
    return image_obj

def get_image_id_from_image_name (conn, image_name, project_id=None, dataset_id=None):

    image_ids = None
    query_service = conn.getQueryService()

    if (dataset_id) :
        # HQL query
        hql_query = """
        SELECT image
        FROM Dataset AS dataset
        JOIN dataset.imageLinks AS image_links
        JOIN image_links.child AS image
        WHERE image.name = :name
        AND dataset.id = :id
        """
        
        # Parameters
        params = ParametersI()
        params.addId(rlong(dataset_id))
        params.addString(f"name", rstring(image_name))

        results = query_service.findAllByQuery(hql_query, params)
        image_ids = [result.id.val for result in results]

    elif (project_id):
        # HQL query by Project
        hql_query = """
        SELECT image
        FROM Project AS project
        JOIN project.datasetLinks AS dataset_links
        JOIN dataset_links.child AS dataset
        JOIN dataset.imageLinks AS image_links
        JOIN image_links.child AS image
        WHERE image.name = :name
        AND project.id = :id
        """
        
        # Parameters
        params = ParametersI()
        params.addId(rlong(project_id))
        params.addString(f"name", rstring(image_name))

        results = query_service.findAllByQuery(hql_query, params)    
        image_ids = [result.id.val for result in results]
    
    return image_ids

def get_key_value_metadata (img_obj):
    map_ann = img_obj.getAnnotation()
    metadata_dict = {k:v for k, v in map_ann.getValue()}
    return metadata_dict

def find_images_in_dataset_by_hql(conn, dataset_id, key, value):
    key = str(key)
    value = str(value)
    query_service = conn.getQueryService()
    
    # HQL query
    hql_query = """
    SELECT image
    FROM Dataset AS dataset
    JOIN dataset.imageLinks AS image_links
    JOIN image_links.child AS image
    JOIN image.annotationLinks AS annotation_links
    JOIN annotation_links.child AS annotation
    JOIN annotation.mapValue as map_value
    WHERE TYPE(annotation) = MapAnnotation
    AND dataset.id = :id
    AND map_value.name = :key
    AND map_value.value = :value
    """

    # Parameters
    params = ParametersI()
    params.addId(rlong(dataset_id))
    params.addString(f"key", rstring(key))
    params.addString(f"value", rstring(value))

    results = query_service.findAllByQuery(hql_query, params)

    return results

def find_image_ids_in_dataset_kv_dict (conn, dataset_id, kv_dict):
    list_of_results = []
    for i, (k, v) in enumerate(kv_dict.items()):
        results = find_images_in_dataset_by_hql(conn, dataset_id, k, v)
        image_ids = [result.id.val for result in results]
        
        list_of_results.append(set(image_ids))
    common_values = list(set.intersection(*list_of_results))
    return common_values