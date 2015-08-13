--[[---------------------------------------------------------------------------

	Module:		Array

	Author:		Mark Swoope

	Date:		August 1, 2015

	Description:	Functions for operating on lua arrays.
			Will expand in the future.

-----------------------------------------------------------------------------]]

require "String"

Array = {}

Array.firstOf = function( table, value )
	for i = 1, #table, 1 do
		if table[i] == value then
			return i
		end
	end
	return nil
end

Array.range = function( min, max )
	local array
	array = {}
	for i = min, max, 1 do
		table.insert( array, i )
	end
	return array
end

Array.ranges = function( str )
	local result, ranges

	result = {}
	ranges = String.split( str, "," )
	if ranges == nil then
		return nil
	end
	for i = 1, #ranges, 1 do
		local range
		range = String.split( ranges[i], "-" )
		if #range == 2 then
			for j = tonumber(range[1]), tonumber(range[2]), 1 do
				table.insert( result, j )
			end
		else
			table.insert( result, tonumber(range[1]) )
		end
	end
	return result
end
