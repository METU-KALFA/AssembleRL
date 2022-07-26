import os
import glob

# path to the asset directory
objects_root = os.path.dirname(__file__)

# load all xmls in assets/objects/
_furniture_xmls = glob.glob(objects_root + "/*.xml")
_furniture_xmls.sort()
_furniture_names = [x.rsplit('/')[-1] for x in _furniture_xmls]
furniture_xmls = [name for name in _furniture_names]
# list of furniture models
furniture_name2id = {
    furniture_name.split('.')[0]: i for i, furniture_name in enumerate(_furniture_names)
}
furniture_names = [furniture_name.split('.')[0] for furniture_name in _furniture_names]
furniture_ids = [i for i in range(len(furniture_names))]

def xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(objects_root, xml_path)
    return full_path