try {
    $mapping = Get-Content "1001 BUILDING FORMS_Dataset_rewrite_value_index.json" -Raw | ConvertFrom-Json
    $reverseMapping = @{}
    foreach ($prop in $mapping.psobject.Properties) {
        if ($prop.Value -is [PSCustomObject]) {
            $innerMap = @{}
            foreach ($innerProp in $prop.Value.psobject.Properties) {
                $innerMap[$innerProp.Value.ToString()] = $innerProp.Name
            }
            $reverseMapping[$prop.Name] = $innerMap
        }
    }

    $scanned = 0
    $modifiedCount = 0
    $unchangedCount = 0
    $warnings = 0
    $samples = @()

    $files = Get-ChildItem -Path "customized_simple_dataset_tagVersion_simplified\data\*\category.json"
    foreach ($file in $files) {
        $scanned++
        $rawContent = Get-Content $file.FullName -Raw
        if (-not $rawContent) { continue }
        $content = $rawContent | ConvertFrom-Json
        $changed = $false
        $before = $rawContent
        
        foreach ($prop in $content.psobject.Properties) {
            $key = $prop.Name
            $val = $prop.Value.ToString()
            
            if ($reverseMapping.ContainsKey($key)) {
                $map = $reverseMapping[$key]
                if ($map.ContainsKey($val)) {
                    $newVal = $map[$val]
                    $content.$key = $newVal
                    $changed = $true
                } else {
                    $isKey = $false
                    foreach ($mapKey in $mapping.$key.psobject.Properties.Name) {
                        if ($mapKey -eq $val) { $isKey = $true; break }
                    }
                    if (-not $isKey) {
                        Write-Host "Warning: No mapping for $key value $val in $($file.FullName)"
                        $warnings++
                    }
                }
            }
        }
        
        if ($changed) {
            $afterObj = $content | ConvertTo-Json
            # Write UTF8 without BOM
            $Utf8NoBomEncoding = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($file.FullName, $afterObj, $Utf8NoBomEncoding)
            
            $modifiedCount++
            if ($samples.Count -lt 3) {
                $samples += [PSCustomObject]@{
                    Path = $file.FullName
                    Before = $before
                    After = $afterObj
                }
            }
        } else {
            $unchangedCount++
        }
    }

    Write-Host "Total scanned: $scanned"
    Write-Host "Files modified: $modifiedCount"
    Write-Host "Files unchanged: $unchangedCount"
    Write-Host "Warnings: $warnings"

    foreach ($s in $samples) {
        Write-Host "`n--- Sample File: $($s.Path) ---"
        Write-Host "Before:"
        Write-Host $s.Before
        Write-Host "After:"
        Write-Host $s.After
    }
} catch {
    Write-Error $_
}
