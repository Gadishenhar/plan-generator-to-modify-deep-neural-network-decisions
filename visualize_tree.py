import copy

def visualize_tree(root):
    """
    Visualizes a monte carlo tree. Prints one full path at a time, from
    root to leaf, with each node's information.
    :param root: The tree's root.
    """
    _visualize_tree(root, [], 0, '-')


def _visualize_tree(root, path_so_far, assigned_idx, parent_idx):
    """
    Internal recursive implementation.
    :param root: The tree's root.
    :param path_so_far: The list of nodes we have seen so far, excluding
    root.
    :param assigned_idx: The unique index we should assign to root.
    """

    # Add the current node to the path list
    path_so_far.append([
        assigned_idx,
        parent_idx,
        root.depth,
        root.num_of_successes,
        root.num_of_passes,
        root.total_cost,
        root.action.action_name,
        root.data
    ])

    # If this node is a leaf, print the full path so far
    if not root.child:
        for node in path_so_far:
            print('-> [ ', ', '.join(list(map(str, node))), ']')
        print()
        return

    # If not, go over all of node's children, ans visualize from them
    for i, child in enumerate(root.child):
        # The index we will assign is a combination of the depth and a unique index
        next_assigned_idx = str(child.depth) + str(i)
        _visualize_tree(child, copy.copy(path_so_far), next_assigned_idx, assigned_idx)

