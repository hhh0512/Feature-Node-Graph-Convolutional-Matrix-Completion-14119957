from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})

    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})
    feed_dict.update({placeholders['class_values']: class_values})

    feed_dict.update({placeholders['dropout']: dropout})

    if u_features_side is not None and v_features_side is not None:
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict


def construct_feed_dict_featnodes(base_feed,
                                  placeholders,
                                  f_features, f_features_nonzero,
                                  support_uf, support_fu, support_vf, support_fv):
    """Extend an existing feed dict with feature-node supports and features."""
    feed = dict(base_feed)
    feed.update({
        placeholders['f_features']: f_features,
        placeholders['f_features_nonzero']: f_features_nonzero,
        placeholders['support_uf']: support_uf,
        placeholders['support_fu']: support_fu,
        placeholders['support_vf']: support_vf,
        placeholders['support_fv']: support_fv,
    })
    return feed
