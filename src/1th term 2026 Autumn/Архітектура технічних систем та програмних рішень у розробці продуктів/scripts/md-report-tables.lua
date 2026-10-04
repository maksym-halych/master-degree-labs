-- Render every table the way the Word template draws them: a black header row
-- with white bold text, and a rule on all four sides of every cell.
--
-- Pandoc's LaTeX writer emits booktabs longtables -- three horizontal rules, no
-- verticals, no shading -- and offers no option to change that. The column
-- specification is written out with the table, so no amount of preamble can add
-- the missing rules after the fact; the table has to be emitted as raw LaTeX
-- instead. Loaded by scripts/render-md-report.sh via --lua-filter.

-- Cell contents are block-level (several of the template's cells hold bullet
-- lists), so each one is written through pandoc's own LaTeX writer rather than
-- flattened to a string.
local function cell_latex(blocks)
    local doc = pandoc.Pandoc(blocks)
    local tex = pandoc.write(doc, 'latex')
    -- Paragraphs inside a p-column cell: a blank line would end the cell's
    -- paragraph and leave vertical space the Word table does not have.
    tex = tex:gsub('%s*\n%s*\n%s*', ' \\par ')
    return (tex:gsub('^%s+', ''):gsub('%s+$', ''))
end

local function row_latex(row, is_header)
    local cells = {}
    for _, cell in ipairs(row.cells) do
        local tex = cell_latex(cell.contents)
        if is_header then
            -- \bfseries after \color so an empty cell still sets both.
            tex = '\\color{white}\\bfseries ' .. tex
        end
        table.insert(cells, tex)
    end
    return table.concat(cells, ' & ') .. ' \\\\'
end

-- Column widths: pandoc supplies them only for grid tables that declare one, so
-- fall back to an even split. The arithmetic subtracts the inter-column padding
-- and rule width that \textwidth does not account for, or the table overruns
-- the margin by a few points per column.
local function colspec_latex(colspecs)
    local n = #colspecs
    local widths = {}
    local total = 0
    for _, spec in ipairs(colspecs) do
        local w = spec[2]
        if type(w) == 'number' and w > 0 then total = total + w end
    end
    for _, spec in ipairs(colspecs) do
        local w = spec[2]
        if type(w) ~= 'number' or w <= 0 or total <= 0 then w = 1 / n end
        table.insert(widths, w)
    end
    local parts = {}
    for _, w in ipairs(widths) do
        table.insert(parts, string.format(
            '>{\\raggedright\\arraybackslash}p{\\dimexpr %.4f\\textwidth-2\\tabcolsep-%.4f\\arrayrulewidth\\relax}',
            w, 1 + 1 / n))
    end
    return '|' .. table.concat(parts, '|') .. '|'
end

function Table(tbl)
    local out = {}
    table.insert(out, '\\begin{longtable}{' .. colspec_latex(tbl.colspecs) .. '}')
    table.insert(out, '\\hline')

    local head_rows = tbl.head.rows
    if #head_rows > 0 then
        for _, row in ipairs(head_rows) do
            table.insert(out, '\\rowcolor{black}')
            table.insert(out, row_latex(row, true))
            table.insert(out, '\\hline')
        end
        -- \endhead repeats the header when a table breaks across pages, as the
        -- Word tables do.
        table.insert(out, '\\endhead')
    end

    for _, body in ipairs(tbl.bodies) do
        for _, row in ipairs(body.body) do
            table.insert(out, row_latex(row, false))
            table.insert(out, '\\hline')
        end
    end

    table.insert(out, '\\end{longtable}')
    return pandoc.RawBlock('latex', table.concat(out, '\n'))
end
