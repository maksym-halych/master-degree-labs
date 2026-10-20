-- Renders one LaTeX text fragment and its reference list in a given CSL style.
--
-- The report shows the same fragment in several citation styles, but biblatex
-- (loaded by the shared preamble) fixes one style per document, and the
-- citation-style-language package cannot be loaded beside it. So this module
-- drives citeproc-lua, the CSL engine that ships with citation-style-language,
-- directly: it reads the same .csl files Zotero uses and the .bib file Zotero
-- exports, and returns plain LaTeX that does not touch biblatex at all.
--
-- In the fragment, \dcite{key} marks a citation. Each one is replaced with the
-- citation the style produces; numeric styles number sources by first use.

local citeproc = require("citeproc")
local bibtex2csl = require("citeproc-bibtex2csl")
local date_node = require("citeproc-node-date")

-- citeproc-lua 0.9.1 joins the parts of a single date with the delimiter of the
-- <date> element in the style, ignoring the one of the localized date format the
-- element defers to. Vancouver defines its date format that way, so its dates
-- came out as "2026Sept23" instead of "2026 Sept 23". Use the delimiter the
-- caller resolved, which is what the range-date code path already does.
local build_single_date = date_node.Date.build_single_date
date_node.Date.build_single_date = function(self, date_parts, single_date, delimiter, ...)
  local ir = build_single_date(self, date_parts, single_date, delimiter, ...)
  ir.delimiter = delimiter
  return ir
end

local M = {}

local function read_file(path, what)
  local handle = io.open(path, "r")
  if not handle then
    error(string.format("csl-demo: cannot open %s '%s'", what, path))
  end
  local contents = handle:read("*a")
  handle:close()
  return contents
end

local function find_and_read(name, what)
  local path = kpse.find_file(name)
  if not path then
    error(string.format("csl-demo: %s '%s' not found", what, name))
  end
  return read_file(path, what)
end

local function load_items(bib_path)
  local items = bibtex2csl.parse_bibtex_to_csl(read_file(bib_path, "bibliography"), true, true, true, true)
  local by_id = {}
  for _, item in ipairs(items) do
    by_id[item.id] = item
  end
  return by_id
end

-- Keys in order of first citation: numeric styles number the list by it.
local function cited_keys(text, items)
  local keys, seen = {}, {}
  for key in text:gmatch("\\dcite{([^}]+)}") do
    if not items[key] then
      error(string.format("csl-demo: no entry '%s' in the bibliography", key))
    end
    if not seen[key] then
      seen[key] = true
      table.insert(keys, key)
    end
  end
  return keys
end

-- Styles added in Zotero are copied into styles/ next to the report; the ones
-- Zotero ships with come from the copies in the TeX distribution.
local function read_style(style_id)
  local local_path = "styles/" .. style_id .. ".csl"
  local handle = io.open(local_path, "r")
  if handle then
    handle:close()
    return read_file(local_path, "style")
  end
  return find_and_read(style_id .. ".csl", "style")
end

-- A dependent style holds only metadata and a link to the style that does the
-- formatting, the way a journal names the house style it follows. Zotero
-- resolves that link itself; the engine needs the parent handed to it.
local function resolve_style(style_id)
  local xml = read_style(style_id)
  local parent = xml:match('<link href="[^"]-/styles/([^"/]+)" rel="independent%-parent"/>')
  if parent then
    texio.write_nl("log", string.format("csl-demo: '%s' is a dependent style of '%s'", style_id, parent))
    return read_style(parent)
  end
  return xml
end

local function make_engine(style_id, items)
  local sys = {
    retrieveLocale = function(lang)
      local path = kpse.find_file("csl-locales-" .. lang .. ".xml")
      return path and read_file(path, "locale")
    end,
    retrieveItem = function(id)
      return items[id]
    end,
  }
  return citeproc.new(sys, resolve_style(style_id))
end

-- citeproc-lua wraps every entry as "\bibitem[label]{id}" for a thebibliography
-- environment, which biblatex redefines. Split the label back out so the list
-- can be typeset with a plain list environment instead.
local function split_entry(entry)
  local label, body = entry:match("^\\bibitem%[(.-)%]{[^}]*}\n(.*)$")
  if label then
    return label:gsub("^{(.*)}$", "%1"), body
  end
  return nil, (entry:gsub("^\\bibitem{[^}]*}\n", ""))
end

local function format_bibliography(entries)
  local labelled = {}
  local widest = ""
  for _, entry in ipairs(entries) do
    local label, body = split_entry(entry)
    body = body:gsub("%s+$", "")
    table.insert(labelled, {label = label, body = body})
    if label and #label > #widest then
      widest = label
    end
  end

  local lines = {}
  if widest ~= "" then
    table.insert(lines, "\\begin{list}{}{\\settowidth{\\labelwidth}{" .. widest .. "}"
      .. "\\setlength{\\leftmargin}{\\dimexpr\\labelwidth+\\labelsep\\relax}\\setlength{\\itemsep}{2pt}}")
    for _, e in ipairs(labelled) do
      table.insert(lines, "\\item[" .. e.label .. "] " .. e.body)
    end
  else
    -- Author-date styles have no labels and use a hanging indent instead.
    table.insert(lines, "\\begin{list}{}{\\setlength{\\leftmargin}{2em}\\setlength{\\itemindent}{-2em}"
      .. "\\setlength{\\labelwidth}{0pt}\\setlength{\\labelsep}{0pt}\\setlength{\\itemsep}{2pt}}")
    for _, e in ipairs(labelled) do
      table.insert(lines, "\\item[] " .. e.body)
    end
  end
  table.insert(lines, "\\end{list}")
  return table.concat(lines, "\n")
end

--- Typesets the fragment and its reference list in the given style.
-- @param style_id CSL style file name without ".csl", looked up in styles/ first, then through kpathsea.
-- @param text_path LaTeX fragment with \dcite{key} citations.
-- @param bib_path BibLaTeX file the keys refer to.
-- @param heading Heading printed above the reference list.
function M.render(style_id, text_path, bib_path, heading)
  local text = read_file(text_path, "text fragment")
  local items = load_items(bib_path)
  local keys = cited_keys(text, items)
  local engine = make_engine(style_id, items)
  engine:updateItems(keys)

  local rendered = text:gsub("\\dcite{([^}]+)}", function(key)
    return engine:makeCitationCluster({{id = key}})
  end)
  local bibliography = engine:makeBibliography()[2]
  texio.write_nl("log", string.format("csl-demo: style '%s', %d citations, %d references",
    style_id, #keys, #bibliography))

  local output = rendered .. "\n\n\\par\\textbf{" .. heading .. "}\n\n" .. format_bibliography(bibliography)
  -- tex.print takes a list of lines, so blank lines in the fragment still end paragraphs.
  local lines = {}
  for line in (output .. "\n"):gmatch("(.-)\n") do
    table.insert(lines, line)
  end
  tex.print(lines)
end

return M
