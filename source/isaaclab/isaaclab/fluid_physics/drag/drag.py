# Attach drag forces to a rigid object / articulation / rigid object collection

# It depends on the velocity of the object, and the velocity of the fluid it's in. It can be in two fluids at once. (e.g. water and air)
# It should be applied at the center of mass of the immerged part of the object.
# It should be modular so that we can have different drag models for different objects.
# It should take into account the shape of the object. We could start with basic shapes like sphere, cylinder, and box.
# Should be able to select the bodies to apply the drag to. I.e. separate the drag from the hull of a ship to its sails.