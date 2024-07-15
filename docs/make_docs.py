"""Script to make environment md"""

import neurogym as ngym
from neurogym.utils.info import info, info_wrapper

SOURCE_ROOT = "https://github.com/gyyang/neurogym/blob/master/"


def add_link(text, link):
    # Add link to a within document location
    return "[{:s}](#{:s})".format(text, link)


def write_doc(write_type):
    all_tags = ngym.all_tags()
    if write_type == "tasks":
        all_items = ngym.all_envs()
        info_fn = info
        fname = "envs.md"
        all_items_dict = ngym.envs.registration.ALL_ENVS

    elif write_type == "wrappers":
        all_items = ngym.all_wrappers()
        info_fn = info_wrapper
        fname = "wrappers.md"
        all_items_dict = ngym.wrappers.ALL_WRAPPERS
    else:
        raise ValueError

    string = ""
    names = ""
    counter = 0
    link_dict = dict()
    for name in all_items:
        # Get information about individual task or wrapper
        string += "___\n\n"
        info_string = info_fn(name)
        info_string = info_string.replace("\n", "  \n")  # for markdown

        # If task, add link to tags
        if write_type == "tasks":
            # Tags has to be last
            ind = info_string.find("Tags")
            info_string = info_string[:ind]
            env = ngym.make(name)
            env = env.unwrapped  # remove extra wrappers ('make' can add OrderEnforcer wrapper, which causes issues here)
            # Modify to add tag links
            info_string += "Tags: "
            for tag in env.metadata.get("tags", []):
                tag_link = tag.lower().replace(" ", "-")
                tag_with_link = add_link(tag, tag_link)
                info_string += tag_with_link + ", "
            info_string = info_string[:-2] + "\n\n"
        string += info_string

        # Make links to the section titles
        # Using github's automatic link to section titles
        if write_type == "tasks":
            env = ngym.make(name)
            env = env.unwrapped  # remove extra wrappers ('make' can add OrderEnforcer wrapper, which causes issues here)
            link = type(env).__name__
        else:
            link = name
        link = link.lower().replace(" ", "-")
        link_dict[name] = link

        # Add link to source code
        names += add_link(name, link) + "\n\n"
        source_link = all_items_dict[name].split(":")[0].replace(".", "/")
        string += "[Source]({:s})\n\n".format(SOURCE_ROOT + source_link + ".py")
        counter += 1

    full_string = "### List of {:d} {:s} implemented\n\n".format(counter, write_type)
    full_string += names

    if write_type == "tasks":
        string_all_tags = "___\n\nTags: "
        for tag in sorted(all_tags):
            tag_link = tag.lower().replace(" ", "-")
            tag_with_link = add_link(tag, tag_link)
            string_all_tags += tag_with_link + ", "
        string_all_tags = string_all_tags[:-2] + "\n\n"
        full_string += string_all_tags

    full_string += string

    if write_type == "tasks":
        string_tag = "___\n\n### Tags ### \n\n"
        for tag in sorted(all_tags):
            string_tag += "### {:s} \n\n".format(tag)
            for name in ngym.all_envs(tag=tag):
                string_tag += add_link(name, link_dict[name])
                string_tag += "\n\n"
        full_string += string_tag

    with open(fname, "w") as f:
        f.write("* Under development, details subject to change\n\n")
        f.write(full_string)


def main():
    write_doc("tasks")
    write_doc("wrappers")


if __name__ == "__main__":
    main()
